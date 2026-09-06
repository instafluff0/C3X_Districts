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
if not defined C3X_LAB_IMPROVEMENTS set "C3X_LAB_IMPROVEMENTS=..\packs\ImprovementsNormalized"
if not defined C3X_LAB_TILE_OBJECTS set "C3X_LAB_TILE_OBJECTS=..\packs\TileObjectsNormalized"
if not defined C3X_LAB_UNITS set "C3X_LAB_UNITS=..\packs\UnitFamilyLab"
if not defined C3X_LAB_COMPOUND_UNITS set "C3X_LAB_COMPOUND_UNITS=..\packs\CompoundUnitLab"
set "C3X_LAB_OUTPUT=..\preview\out\terrain_lab"
set "C3X_LAB_BIQ_VIEW=%C3X_LAB_OUTPUT%\test_biq_l13_rivers_192.csv"
set "C3X_LAB_ROAD_SCENARIO=fixtures\l14_roads_192.csv"
set "C3X_LAB_RAILROAD_SCENARIO=fixtures\l15_railroads_192.csv"
set "C3X_LAB_RESOURCE_SCENARIO=fixtures\l16_resources_192.csv"
set "C3X_LAB_CITY_SCENARIO=fixtures\l17_cities_192.csv"
set "C3X_LAB_MINE_SCENARIO=fixtures\l18_mines_192.csv"
set "C3X_LAB_FARM_SCENARIO=fixtures\l19_farms_192.csv"
set "C3X_LAB_TILE_OBJECT_SCENARIO=fixtures\l19a_tile_objects_192.csv"
set "C3X_LAB_INFRASTRUCTURE_SCENARIO=fixtures\l19b_infrastructure_192.csv"
set "C3X_LAB_UNIT_SCENARIO=fixtures\l20_units_192.csv"
if not exist "%C3X_LAB_OUTPUT%" mkdir "%C3X_LAB_OUTPUT%"

for %%F in (unit_archer_runtime.bin unit_swordsman_runtime.bin unit_infantry_runtime.bin unit_fighter_runtime.bin unit_galley_runtime.bin unit_worker_runtime.bin) do if not exist "%C3X_LAB_UNITS%\%%F" goto missing
for %%F in (unit_horseman_runtime.bin unit_catapult_runtime.bin unit_tank_runtime.bin unit_great_general_classical_runtime.bin) do if not exist "%C3X_LAB_COMPOUND_UNITS%\%%F" goto missing
if not exist "%C3X_LAB_UNIT_SCENARIO%" goto missing

set "C3X_L21_MODES=noon sunset midnight sunrise zoom2 no_units no_borders"
if not "%~1"=="" set "C3X_L21_MODES=%~1"
for %%M in (%C3X_L21_MODES%) do (
  set "C3X_LAB_RENDER_OK="
  for /l %%R in (1,1,3) do if not defined C3X_LAB_RENDER_OK (
    build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l21_complete_%%M.bmp" beauty_complete_%%M 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%" "%C3X_LAB_SHORE_FEATURES%" "%C3X_LAB_ROUTE_STYLES%" "%C3X_LAB_ROAD_SCENARIO%" "%C3X_LAB_ROUTE_DOODADS%" "%C3X_LAB_RAILROAD_SCENARIO%" "%C3X_LAB_RESOURCES%" "%C3X_LAB_RESOURCE_SCENARIO%" "%C3X_LAB_CITIES%" "%C3X_LAB_CITY_ADJUNCTS%" "%C3X_LAB_CITY_SCENARIO%" "%C3X_LAB_IMPROVEMENTS%" "%C3X_LAB_MINE_SCENARIO%" "%C3X_LAB_IMPROVEMENTS%" "%C3X_LAB_FARM_SCENARIO%" "%C3X_LAB_TILE_OBJECTS%" "%C3X_LAB_TILE_OBJECT_SCENARIO%" "%C3X_LAB_TILE_OBJECTS%" "%C3X_LAB_INFRASTRUCTURE_SCENARIO%" "%C3X_LAB_UNITS%" "%C3X_LAB_COMPOUND_UNITS%" "%C3X_LAB_UNIT_SCENARIO%"
    if not errorlevel 1 set "C3X_LAB_RENDER_OK=1"
    if errorlevel 1 ping -n 16 127.0.0.1 >nul
  )
  if not defined C3X_LAB_RENDER_OK goto failed
  rem Give the VM D3D driver time to retire the prior 3200x1800 process.
  ping -n 6 127.0.0.1 >nul
)

popd
exit /b 0
:failed
popd
exit /b 1
:missing
echo terrain_lab: L21 requires all approved alternate-skin L9-L20 packs and scenarios
popd
exit /b 1
