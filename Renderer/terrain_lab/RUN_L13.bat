@echo off
setlocal
pushd "%~dp0"

if exist "local_paths.bat" call local_paths.bat

call PREPARE_L13_BIQ_VIEWPORT.bat
if errorlevel 1 goto failed
call PREPARE_L13_RIVER_TOPOLOGY_VIEWPORT.bat
if errorlevel 1 goto failed
call BUILD.bat
if errorlevel 1 goto failed

if not defined C3X_LAB_PACK set "C3X_LAB_PACK=..\packs\Civ5EnvironmentSkin"
if not defined C3X_LAB_VEGETATION set "C3X_LAB_VEGETATION=..\packs\Civ5EnvironmentVegetation"
if not defined C3X_LAB_DECALS set "C3X_LAB_DECALS=..\packs\DecalsNormalized"
if not defined C3X_LAB_TERRAIN_ELEMENTS set "C3X_LAB_TERRAIN_ELEMENTS=..\packs\TerrainElementsNormalized"
if not defined C3X_LAB_SHORE_FEATURES set "C3X_LAB_SHORE_FEATURES=..\packs\ShoreNormalized"
set "C3X_LAB_OUTPUT=..\preview\out\terrain_lab"
set "C3X_LAB_BIQ_VIEW=%C3X_LAB_OUTPUT%\test_biq_l13_rivers_192.csv"
set "C3X_LAB_BIQ_TOPOLOGY_VIEW=%C3X_LAB_OUTPUT%\test_biq_l13_river_topology_192.csv"
if not exist "%C3X_LAB_OUTPUT%" mkdir "%C3X_LAB_OUTPUT%"

if not exist "%C3X_LAB_PACK%\manifest.json" goto missing_alternate
if not exist "%C3X_LAB_VEGETATION%\vegetation_runtime.bin" goto missing_alternate
if not exist "%C3X_LAB_SHORE_FEATURES%\shore_runtime.bin" goto missing_shore

build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l13_rivers.bmp" beauty_rivers 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%" "%C3X_LAB_SHORE_FEATURES%"
if errorlevel 1 goto failed
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l13_rivers_no_rivers.bmp" beauty_rivers_no_rivers 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%" "%C3X_LAB_SHORE_FEATURES%"
if errorlevel 1 goto failed
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l13_rivers_only.bmp" beauty_rivers_only 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%" "%C3X_LAB_SHORE_FEATURES%"
if errorlevel 1 goto failed
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l13_rivers_thumbnail.bmp" beauty_rivers_thumbnail 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%" "%C3X_LAB_SHORE_FEATURES%"
if errorlevel 1 goto failed
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l13_river_topology.bmp" beauty_rivers 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_TOPOLOGY_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%" "%C3X_LAB_SHORE_FEATURES%"
set "C3X_LAB_RESULT=%errorlevel%"
popd
exit /b %C3X_LAB_RESULT%

:failed
popd
exit /b 1

:missing_alternate
echo terrain_lab: L13 requires the local alternate environment terrain and vegetation packs
popd
exit /b 1

:missing_shore
echo terrain_lab: L13 requires the normalized shore runtime bundle
popd
exit /b 1
