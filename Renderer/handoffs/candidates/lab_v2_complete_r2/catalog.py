"""Explicit selection policy; revision numbers alone never select an experiment."""
V2 = 'Renderer/terrain_lab/v2/'
BEAUTY = V2+'audits/beauty/'

# id, legacy gates, native disposition, selected implementation, remaining work
SYSTEMS = [
 ('terrain', ['L9','L10','L11','L12'], 'pickup_r1_integrated', 'shadow-receiver-r1 / relief-size-r3 / coast-pass-rocks-r8', 'Keep current native performance fixes; analytic dunes and incomplete ground-layer reconstruction remain unaccepted.'),
 ('environment', ['L13A'], 'integrated_with_shadow_basis_gap', 'shared captured EnvironmentState; Q6 source-triangle field; scene-linear composition', 'Unify posed object coordinate/normal/height bases; preserve retained UI outside exposure/glow.'),
 ('rivers', ['L13'], 'older_rivers_integrated', 'river-corridor-r3 over r2', 'Port corridor, headwater and bank-rock recipe together; pool/bank fidelity and multi-height water remain open.'),
 ('forest_jungle', ['L9','L12'], 'older_canopy_integrated', 'canopy-variation-r1', 'Port stable layout seeds and exclusions with the river corridor; no per-frame reshuffling.'),
 ('water', ['L9','L13A'], 'older_water_integrated', 'water-natural-r6 + GPU water-reflection-r5', 'Port linear reflection prepass, offscreen halo and material bindings; animation/surf deferred by user.'),
 ('roads', ['L14'], 'integrated', 'preserve L14 and current native implementation', 'No new road design in this pickup; adapt corridor/bridge clearance when composing hydrology.'),
 ('railroads', ['L15'], 'integrated', 'preserve L15 and current native implementation', 'Preserve worked/pillaged bridges and exact shared edges.'),
 ('resources', ['L16'], 'static_integrated_animation_maintenance_staged', 'current static source caster plus current 10-family animation runtime', 'Animated bodies receive static shadows but are excluded as casters; add posed self/cast shadows with bounded dynamic receivers.'),
 ('cities', ['L17'], 'older_city_kit_integrated', 'central American r111/r112; conditional style selections in cases', 'Port single-era layout/material/ground/light chain; coastal surrounded fit and complete style/size coverage remain open.'),
 ('palaces', ['L17'], 'lab_only_new_library_and_layout', '47 standard roots; authoritative capital; opaque style fallback IDs', 'Central grid-aligned placement r111/r112 only. Four Gran Colombian Tree_B_Lg children unresolved; scenario bindings excluded.'),
 ('city_walls', ['L17'], 'integrated_legacy', 'L17 normalized wall runtime', 'Reconcile wall envelope with new central/paving footprints; not proven by the current no-wall central witnesses.'),
 ('mines', ['L18'], 'integrated', 'L18 recursive source parts, ground and emissive channels', 'Preserve source compound transforms; migrate shared material/light interfaces without restoring legacy blob shadows.'),
 ('farms_irrigation_tundra', ['L19'], 'native_code_present_formal_I19_pending', 'L19 farm runtime and independent tundra material', 'Full source/era/state composition with new terrain; presence in a pickup smoke image is not gate promotion.'),
 ('goody_huts_colonies', ['L19A'], 'not_integrated', 'L19A tile_object_runtime.bin, explicit village/colony semantics', 'Viewer-visible known objects only; owner/era/resource transitions; resolve frozen L19A asset hash mismatch explicitly.'),
 ('fortress_barricade', ['L19B'], 'not_integrated', 'L19B fortification_runtime.bin', 'Authoritative state selects one complete kit and shadow; no invented Districts.'),
 ('airfield', ['L19B'], 'not_integrated', 'L19B airfield_runtime.bin', 'Preserve ground and structure ordering, visibility/removal, source alpha and shared shadows.'),
 ('outpost_radar_victory', ['L19B'], 'not_integrated', 'L19B ground_state_runtime.bin raised objects', 'Authoritative object states and owner coloring; migrate source-mesh cast shadows.'),
 ('pollution_craters', ['L19B'], 'not_integrated', 'L19B ground_state_runtime.bin flat state art', 'Depth-conforming non-casters; coexist with resources/routes, clear on state removal.'),
 ('territory_borders', ['L21'], 'retained_native_overlay', 'L21 add_territory_scene and main-color ribbons', 'Lab ownership grid is synthetic. Capture authoritative per-tile owners and define exclusive retained/custom overlay ownership before enabling.'),
 ('units', ['L20'], 'animation_maintenance_staged_formal_I20_pending', 'current native 9 families / 79 actions, generic normalized poses/tools', 'Keep action cursor, dirty bounds and native background. Reconcile direction and add relief/object/cross-unit receivers plus alpha-aware casters.'),
 ('effects', ['L20'], 'partial_preparation_not_global_visual_coverage', 'generic effect graphs, owner-color/action/compound contracts', 'Preserve legacy L20 and offline sources; no claim of complete compound units, combat VFX or attached particle transport in current 9-family runtime.'),
 ('barbarian_camps', [], 'offline_only_not_promoted', 'VIL_BAR_01/VIL_BAR_IND converted offline to generic IDs', 'Dedicated Lab promotion required; not a goody hut or an implied L19A extension.'),
 ('combined_scene', ['L21'], 'formal_I21_pending', 'selected 100-tile natural/city witnesses plus historical 192-tile object composition', 'No single all-new-system production image exists yet; combined visual checkpoint required after staged composition.'),
 ('wonders_districts', [], 'deferred_M9_M10_M11', 'existing future contracts only', 'Do not implement early.'),
]

ENTRIES = {
 'terrain': ['shared/frozen_scene.cpp','shared/recording_adapter.h','systems/terrain/surface.h','systems/terrain/scene_adapter.h'],
 'environment': ['systems/lighting/shadow_field_v1.h','systems/lighting/alpha_coverage_v1.h','shaders/lighting/shadow_visibility_v1.hlsl','shaders/common/hdr_glow_tiled.hlsl'],
 'rivers': ['systems/hydrology/river_corridor.h','systems/hydrology/field.h','systems/hydrology/scene_adapter.h','qa/river_corridor_pass.py'],
 'forest_jungle': ['systems/objects/canopy_layout.h','qa/canopy_variation_pass.py'],
 'water': ['shaders/hydrology/water_natural.hlsl','shaders/hydrology/planar_reflection_pass.hlsl','qa/water_reflection_pass.py'],
 'cities': ['qa/city_central_capital_probe.py','qa/city_scene_pass.py','systems/objects/presentation.py','systems/objects/city_growth_layout.py','systems/objects/city_generator_layout.py','systems/objects/settlement_ground.py','systems/objects/city_ground_geometry.py','qa/city_light_buffer_probe.py','qa/city_facade_light_probe.py','qa/settlement_ground_probe.py','qa/city_environment_probe.py','shaders/objects/city_scene_material.hlsl','shaders/objects/settlement_ground.hlsl','shaders/lighting/local_facade_lights.hlsl','shaders/lighting/city_environment.hlsl'],
 'palaces': ['systems/objects/capital_styles.json'],
 'territory_borders': ['shared/frozen_scene.cpp','shaders/common/frozen_l21.hlsl'],
}

AUDITS = ['CURRENT_VISUAL.md','SHADOW_RECEIVER_PASS.md','SHADOW_RECEIVER_r1_EVIDENCE.json',
 'RELIEF_SIZE_PASS.md','COAST_SOURCE_JOIN_PASS.md','RIVER_VEGETATION_CAMPAIGN.md',
 'RIVER_VEGETATION_PASS_r2.md','RIVER_BANK_ROCK_PASS_r3.md','RIVER_BANK_ROCK_r3_EVIDENCE.json',
 'CANOPY_VARIATION_r1_EVIDENCE.json','WATER_EFFECTS_EXPLORATION.md','WATER_OBJECT_REFLECTIONS.md',
 'WATER_NATURAL_r6_EVIDENCE.json','WATER_REFLECTION_r5_EVIDENCE.json',
 'SURFACE_RICHNESS_CAMPAIGN.md','GROUND_LAYER_FINDINGS.md','CITY_QUALITY_CAMPAIGN.md',
 'CITY_CENTRAL_CAPITAL_PASS.md','CITY_CENTRAL_CAPITAL_EVIDENCE.json',
 'CITY_CAPITAL_MATERIAL_PASS.md','CITY_CAPITAL_MATERIAL_EVIDENCE.json',
 'CITY_PALACE_GROUND_ALIGNMENT_PASS.md','CITY_PALACE_GROUND_ALIGNMENT_EVIDENCE.json',
 'CITY_PALACE_FACADE_ALIGNMENT_PASS.md','CITY_PALACE_FACADE_ALIGNMENT_EVIDENCE.json',
 'CITY_PALACE_COMPOSITION_PASS.md','CITY_PALACE_COMPOSITION_EVIDENCE.json',
 'CITY_ENVIRONMENT_PASS.md','CITY_ENVIRONMENT_EVIDENCE.json','CITY_LIGHT_BUFFER_PASS.md',
 'CITY_SOURCE_SURFACE_PASS.md','CITY_EXTRA_MATERIAL_PASS.md','CITY_GROWTH_HIERARCHY_PASS.md']

# Each case is a selected conditional witness, not an instruction to mix layouts.
CASES = []
for region in ['coastal','inland','wilderness','longcoast','freshcoast','freshrelief','combinedvolcano','freshshadow']:
    CASES.append(('terrain-'+region, 'shadow-receiver-r1/'+region, 'retained terrain foundation; synthetic only for combinedvolcano'))
for region in ['coastal','inland','wilderness','freshcanopy']:
    CASES.append(('river-'+region, 'river-corridor-r3/'+region, 'river/canopy combined candidate'))
for region in ['coastal','inland','wilderness','longcoast','freshwater']:
    CASES.append(('water-'+region, 'water-reflection-r5/'+region+'/combined', 'r6 natural water plus GPU reflection'))
for name,folder,note in [
 ('capital-inland','city-central-capital-r2/inland/environment/render','r111 central orthogonal modern'),
 ('capital-holdout','city-central-capital-r2/holdout/environment/render','r112 central orthogonal modern; freshcanopy'),
 ('capital-coast-fallback','city-palace-facade-alignment-r1/environment/render','r101 corrected material/paving/light; DOES NOT meet central preference'),
 ('asian-small','city-palace-composition-r2/asian-small/render','r93 single-era palace; central preference not applied'),
 ('asian-medium','city-palace-composition-r2/asian-medium/render','r94 single-era palace; central preference not applied'),
 ('asian-large','city-palace-composition-r2/asian-large/render','r92 single-era palace; central preference not applied'),
 ('ancient-inland','city-palace-composition-r1/ancient-medium/render','r77 single-era palace; central preference not applied'),
 ('ancient-coast','city-palace-composition-r2/ancient-coast/render','r98 single-era palace; central preference not applied'),
 ('modern-inland','city-environment-r2/inland-large/render','provisional modern ordinary-city environment'),
 ('modern-wilderness','city-environment-r2/wilderness-medium/render','provisional modern ordinary-city environment')]:
    CASES.append((name,folder,note))

EXCLUDED = [
 'Mixed-era city r17; current preference is one era per city.',
 'Newest surface-richness/continental bake/ground decal diagnostics are NOT selected defaults.',
 'Asian environment roof trial is rejected; retain preceding palace material appearance.',
 'Failed coastal central layouts r106-r110/r113-r115, alternate r116, uncaptured r117.',
 'Unapproved analytic dune geometry is not rehabilitated by this catalog.',
 'Source-only camps, analytic lights, particle bindings and scenario palaces are not silently enabled.',
 'Legacy L21 border frame hashes are pre-territory; they do not validate the final border revision.',
]

# The current Lab state of the art is intentionally a set of isolated witnesses.
# It does not select the older combined-scene fixture, whose composition predates
# the source-fidelity corrections below.
STATE_OF_ART_STUDIES = [
    {
        'id': 'mountain',
        'fixture': V2+'fixtures/relief/beauty-mountain.fixture.json',
        'module': V2+'systems/relief/beauty_mountain.module.json',
        'source': V2+'systems/relief/beauty_mountain.cpp',
        'shader': V2+'shaders/relief/beauty_mountain.hlsl',
        'audit': V2+'audits/relief/BEAUTY_MOUNTAIN_STUDY.md',
        'report': V2+'audits/relief/out/beauty-mountain-civ5-r4/report.json',
        'review_image': V2+'audits/relief/out/beauty-mountain-civ5-r4/h12-z1-pan00.png',
        'review_sha256': 'c3e0329aa617255fd144abe930b34aad413fed66b7d3caac209c355709722429',
        'raw_sha256': '3ac8808024a27ed6f644dd7046ca2cc9e1793e1af216d0f12ff0184517de59e1',
        'disposition': 'user_retained',
    },
    {
        'id': 'grass_plains_tundra_hills',
        'fixture': V2+'fixtures/relief/beauty-terrain.fixture.json',
        'module': V2+'systems/relief/beauty_terrain.module.json',
        'source': V2+'systems/relief/beauty_terrain.cpp',
        'shader': V2+'shaders/relief/beauty_terrain.hlsl',
        'audit': V2+'audits/relief/CIV5_TERRAIN_SOURCE_PASS.md',
        'report': V2+'audits/relief/out/beauty-land-types-r1/report.json',
        'review_image': V2+'audits/relief/out/beauty-land-types-r1/h12-z1-pan00-civ5-lut.png',
        'review_sha256': '8892fefd1ee4c2aead09a027289269741a91655b11d7cf865925e04b8db6339a',
        'raw_sha256': '2be5cd230b0d70ec07ebb0a4b7cc00b913136a01f0d423cf3475fa2f55109215',
        'disposition': 'agent_visual_qa_pass',
    },
    {
        'id': 'forest',
        'fixture': V2+'fixtures/objects/beauty-trees.fixture.json',
        'module': V2+'systems/objects/beauty_objects.module.json',
        'source': V2+'systems/objects/beauty_objects.cpp',
        'shader': V2+'shaders/objects/beauty_objects.hlsl',
        'audit': V2+'audits/objects/CIV5_TREE_SOURCE_PASS.md',
        'report': V2+'audits/objects/out/beauty-trees-r18/report.json',
        'review_image': V2+'audits/objects/out/beauty-trees-r18/h12-z1-pan00-civ5-lut.png',
        'review_sha256': 'fbd3c494d19d16c7becf7729db7d0c4e993eb788262ec6ad10ad1131ecf32bc4',
        'raw_sha256': '36365a40cdb66430d624cb8af188f0bd04ff285f8c87c974570e14d276ddc209',
        'disposition': 'user_good_for_now',
    },
    {
        'id': 'warrior',
        'fixture': V2+'fixtures/objects/beauty-warrior.fixture.json',
        'module': V2+'systems/objects/beauty_objects.module.json',
        'source': V2+'systems/objects/beauty_objects.cpp',
        'shader': V2+'shaders/objects/beauty_objects.hlsl',
        'audit': V2+'audits/objects/WARRIOR_SOURCE_PASS.md',
        'report': V2+'audits/objects/out/beauty-warrior-r4/report.json',
        'review_image': V2+'audits/objects/out/beauty-warrior-r4/h12-z1-pan00-civ5-lut.png',
        'review_sha256': 'e90022920d1317fc113051475280d51fc4222ca37324acca898a58f4e8111d1a',
        'raw_sha256': '35e3930cfa63fdbbc442a92644f254c5986e2e1627541e4ba38e835bdc82736d',
        'disposition': 'user_good_for_now',
    },
]
