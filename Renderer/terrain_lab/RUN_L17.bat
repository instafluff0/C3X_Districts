@echo off
setlocal
pushd "%~dp0"

if exist "local_paths.bat" call local_paths.bat
call PREPARE_L13_BIQ_VIEWPORT.bat
if errorlevel 1 goto failed
call BUILD.bat
if errorlevel 1 goto failed

if not defined C3X_LAB_PACK set "C3X_LAB_PACK=..\packs\Civ5EnvironmentSkin"
if not defined C3X_LAB_VEGETATION set "C3X_LAB_VEGETATION=..\packs\Civ5EnvironmentVegetation"
if not defined C3X_LAB_DECALS set "C3X_LAB_DECALS=..\packs\DecalsNormalized"
if not defined C3X_LAB_TERRAIN_ELEMENTS set "C3X_LAB_TERRAIN_ELEMENTS=..\packs\TerrainElementsNormalized"
if not defined C3X_LAB_SHORE_FEATURES set "C3X_LAB_SHORE_FEATURES=..\packs\ShoreNormalized"
if not defined C3X_LAB_ROUTE_STYLES set "C3X_LAB_ROUTE_STYLES=..\packs\RouteStylesNormalized"
if not defined C3X_LAB_ROUTE_DOODADS set "C3X_LAB_ROUTE_DOODADS=..\packs\RouteDoodadsNormalized"
if not defined C3X_LAB_RESOURCES set "C3X_LAB_RESOURCES=..\packs\ResourceNormalized"
if not defined C3X_LAB_CITIES set "C3X_LAB_CITIES=..\packs\CityComponentsNormalized"
if not defined C3X_LAB_CITY_ADJUNCTS set "C3X_LAB_CITY_ADJUNCTS=..\packs\CityAdjunctsNormalized"
set "C3X_LAB_OUTPUT=..\preview\out\terrain_lab"
set "C3X_LAB_BIQ_VIEW=%C3X_LAB_OUTPUT%\test_biq_l13_rivers_192.csv"
set "C3X_LAB_ROAD_SCENARIO=fixtures\l14_roads_192.csv"
set "C3X_LAB_RAILROAD_SCENARIO=fixtures\l15_railroads_192.csv"
set "C3X_LAB_RESOURCE_SCENARIO=fixtures\l16_resources_192.csv"
set "C3X_LAB_CITY_SCENARIO=fixtures\l17_cities_192.csv"
if not exist "%C3X_LAB_OUTPUT%" mkdir "%C3X_LAB_OUTPUT%"

if not exist "%C3X_LAB_CITIES%\city_runtime.bin" goto missing
if not exist "%C3X_LAB_CITY_ADJUNCTS%\wall_runtime.bin" goto missing
if not exist "%C3X_LAB_CITY_SCENARIO%" goto missing

for %%M in (cities_noon cities_night cities_zoom2 cities_no_cities cities_only) do (
  build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l17_%%M.bmp" beauty_%%M 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%" "%C3X_LAB_SHORE_FEATURES%" "%C3X_LAB_ROUTE_STYLES%" "%C3X_LAB_ROAD_SCENARIO%" "%C3X_LAB_ROUTE_DOODADS%" "%C3X_LAB_RAILROAD_SCENARIO%" "%C3X_LAB_RESOURCES%" "%C3X_LAB_RESOURCE_SCENARIO%" "%C3X_LAB_CITIES%" "%C3X_LAB_CITY_ADJUNCTS%" "%C3X_LAB_CITY_SCENARIO%"
  if errorlevel 1 goto failed
)

popd
exit /b 0
:failed
popd
exit /b 1
:missing
echo terrain_lab: L17 requires normalized city/wall runtime bundles and deterministic city scenario
popd
exit /b 1
