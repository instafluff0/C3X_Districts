"""Explicit selection policy; revision numbers alone never select an experiment."""
V2 = 'Renderer/terrain_lab/v2/'
BEAUTY = V2+'audits/beauty/'

# id, legacy gates, native disposition, selected implementation, remaining work
SYSTEMS = [
 ('terrain', ['L9','L10','L11','L12'], 'pickup_r2_ready_for_game_integration', 'source-fidelity-r13 composes exact beauty_terrain + beauty_mountain providers at 4x/16x sampling with corrected shared world projection and retained hydrology', 'Port the r2 viewport adapters, generic pack bindings and exact canonical world-to-screen basis; analytic dunes and tundra snow-hill decals remain unaccepted.'),
 ('environment', ['L13A'], 'integrated_with_remaining_animated_subject_gap', 'one shared captured EnvironmentState; ShadowL drives face lighting and Q6 source-triangle projection; scene-linear composition', 'Preserve the single light vector and canonical world basis for all static natural casters; reconcile only still-pending posed animated subjects and keep retained UI outside exposure/glow.'),
 ('rivers', ['L13'], 'older_rivers_integrated', 'river-corridor-r3 over r2', 'Port corridor, headwater and bank-rock recipe together; pool/bank fidelity and multi-height water remain open.'),
 ('forest_jungle', ['L9','L12'], 'pickup_r2_ready_for_game_integration', 'BeautyStudies forest path: 22 source bodies / 25 ArtDef recipes / weight 180 / authored opacity, packed normals, readable key/fill face separation and opacity-aware source-mesh directional shadows', 'Port stable source-recipe scatter, material addressing, shared ShadowL face/cast lighting and authoritative building/river/coastline exclusions.'),
 ('water', ['L9','L13A'], 'older_water_integrated', 'water-natural-r6 + GPU water-reflection-r5', 'Port linear reflection prepass, offscreen halo and material bindings; animation/surf deferred by user.'),
 ('roads', ['L14'], 'integrated', 'preserve L14 and current native implementation', 'No new road design in this pickup; adapt corridor/bridge clearance when composing hydrology.'),
 ('railroads', ['L15'], 'integrated', 'preserve L15 and current native implementation', 'Preserve worked/pillaged bridges and exact shared edges.'),
 ('resources', ['L16'], 'static_integrated_animation_maintenance_staged', 'current static source caster plus current 10-family animation runtime', 'Animated bodies receive static shadows but are excluded as casters; add posed self/cast shadows with bounded dynamic receivers.'),
 ('cities', ['L17'], 'older_city_kit_integrated_unchanged_by_source_fidelity_update', 'preserve existing r2 city selections exactly', 'No city visual work is selected here. Later composition must use city footprints as forest exclusion masks; coastal fit and complete style/size coverage remain pre-existing open work.'),
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
 ('units', ['L20'], 'animation_maintenance_staged_formal_I20_pending', 'current native animation plus Warrior source-material correction from beauty_objects', 'Keep action cursor, dirty bounds and native background. Port packed normals and per-material repeat/clamp addressing; retain uniform XYZ scale and reconcile shadows.'),
 ('effects', ['L20'], 'partial_preparation_not_global_visual_coverage', 'generic effect graphs, owner-color/action/compound contracts', 'Preserve legacy L20 and offline sources; no claim of complete compound units, combat VFX or attached particle transport in current 9-family runtime.'),
 ('barbarian_camps', [], 'offline_only_not_promoted', 'VIL_BAR_01/VIL_BAR_IND converted offline to generic IDs', 'Dedicated Lab promotion required; not a goody hut or an implied L19A extension.'),
 ('combined_scene', ['L21'], 'formal_I21_pending_game_integration', 'source-fidelity-r13 is the accepted 100-tile natural-scene pickup; exact terrain/mountain providers, one face/cast light vector, source-mesh tree shadows and retained hydrology coexist', 'Port the selected natural-scene paths and executable shadow contract into the game renderer. Cities remain unchanged and must supply authoritative tree-exclusion footprints.'),
 ('wonders_districts', [], 'deferred_M9_M10_M11', 'existing future contracts only', 'Do not implement early.'),
]

ENTRIES = {
 'terrain': ['systems/relief/beauty_terrain.module.json','systems/relief/beauty_terrain.cpp','shaders/relief/beauty_terrain.hlsl','systems/relief/beauty_mountain.module.json','systems/relief/beauty_mountain.cpp','shaders/relief/beauty_mountain.hlsl','fixtures/beauty/source-fidelity-r2/inland/terrain.hlsl','fixtures/beauty/source-fidelity-r2/inland/mountain.hlsl','fixtures/beauty/source-fidelity-r2/inland/terrain.module.json','fixtures/beauty/source-fidelity-r2/inland/mountain.module.json','fixtures/beauty/source-fidelity-r2/inland/hydrology.module.json','shaders/common/frozen_l21.hlsl','shared/frozen_scene.cpp','shared/recording_adapter.h','systems/terrain/surface.h','systems/terrain/scene_adapter.h'],
 'environment': ['systems/lighting/scene_shadow.cpp','systems/lighting/shadow_field_v1.h','systems/lighting/alpha_coverage_v1.h','shaders/lighting/shadow_visibility_v1.hlsl','shaders/common/hdr_glow_tiled.hlsl','tests/lighting/shadow_contract.cpp'],
 'rivers': ['systems/hydrology/river_corridor.h','systems/hydrology/field.h','systems/hydrology/scene_adapter.h','qa/river_corridor_pass.py'],
 'forest_jungle': ['fixtures/beauty/source-fidelity-r2/inland/forest.hlsl','fixtures/beauty/source-fidelity-r2/inland/forest.module.json','systems/objects/beauty_objects.cpp','shaders/objects/beauty_objects.hlsl','systems/objects/canopy_layout.h','systems/lighting/alpha_coverage_v1.h','qa/canopy_variation_pass.py'],
 'water': ['shaders/hydrology/water_natural.hlsl','shaders/hydrology/planar_reflection_pass.hlsl','qa/water_reflection_pass.py'],
 'cities': ['qa/city_central_capital_probe.py','qa/city_scene_pass.py','systems/objects/presentation.py','systems/objects/city_growth_layout.py','systems/objects/city_generator_layout.py','systems/objects/settlement_ground.py','systems/objects/city_ground_geometry.py','qa/city_light_buffer_probe.py','qa/city_facade_light_probe.py','qa/settlement_ground_probe.py','qa/city_environment_probe.py','shaders/objects/city_scene_material.hlsl','shaders/objects/settlement_ground.hlsl','shaders/lighting/local_facade_lights.hlsl','shaders/lighting/city_environment.hlsl'],
 'palaces': ['systems/objects/capital_styles.json'],
 'territory_borders': ['shared/frozen_scene.cpp','shaders/common/frozen_l21.hlsl'],
 'units': ['systems/objects/beauty_objects.cpp','shaders/objects/beauty_objects.hlsl'],
 'combined_scene': ['fixtures/beauty/source-fidelity-r2/inland/fixture.json','fixtures/beauty/source-fidelity-r2/inland-shadow-control/fixture.json','fixtures/beauty/source-fidelity-r2/inland/base.hlsl','qa/source_fidelity_pickup_evidence.py','qa/source_fidelity_shadow_evidence.py'],
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

# Isolated witnesses remain the per-system source-of-truth. The composed witness
# below proves that the selected natural-system paths coexist on the same 100-tile
# viewport; it does not replace the per-system evidence or authorize city changes.
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
        'report': V2+'audits/objects/out/beauty-trees-r19/report.json',
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
        'report': V2+'audits/objects/out/beauty-warrior-r5/report.json',
        'review_image': V2+'audits/objects/out/beauty-warrior-r4/h12-z1-pan00-civ5-lut.png',
        'review_sha256': 'e90022920d1317fc113051475280d51fc4222ca37324acca898a58f4e8111d1a',
        'raw_sha256': '35e3930cfa63fdbbc442a92644f254c5986e2e1627541e4ba38e835bdc82736d',
        'disposition': 'user_good_for_now',
    },
]

COMPOSED_STATE_OF_ART = {
    'id': 'natural_scene_100_tile',
    'fixture': V2+'fixtures/beauty/source-fidelity-r2/inland/fixture.json',
    'terrain_module': V2+'fixtures/beauty/source-fidelity-r2/inland/terrain.module.json',
    'mountain_module': V2+'fixtures/beauty/source-fidelity-r2/inland/mountain.module.json',
    'forest_module': V2+'fixtures/beauty/source-fidelity-r2/inland/forest.module.json',
    'hydrology_module': V2+'fixtures/beauty/source-fidelity-r2/inland/hydrology.module.json',
    'terrain_shader': V2+'fixtures/beauty/source-fidelity-r2/inland/terrain.hlsl',
    'mountain_source': V2+'systems/relief/beauty_mountain.cpp',
    'mountain_shader': V2+'fixtures/beauty/source-fidelity-r2/inland/mountain.hlsl',
    'forest_source': V2+'systems/objects/beauty_objects.cpp',
    'forest_shader': V2+'fixtures/beauty/source-fidelity-r2/inland/forest.hlsl',
    'audit': V2+'audits/beauty/SOURCE_FIDELITY_PICKUP.md',
    'report': V2+'audits/beauty/out/source-fidelity-r13/inland/report.json',
    'repeat_report': V2+'audits/beauty/out/source-fidelity-r14/inland/report.json',
    'raw_image': V2+'audits/beauty/out/source-fidelity-r13/inland/h12-z1-pan00.bmp',
    'raw_sha256': 'd611be1d83d440a2284ca7e91c366fc4ad14bc4833efda8293f22f7c7008ee7c',
    'review_image': V2+'audits/beauty/out/source-fidelity-r13/inland/h12-z1-pan00.png',
    'review_sha256': '82e7f85e3788f2995e95616a7ca906a72ce976948a3e15b9e10d4ab842a38f2e',
    'comparison': V2+'audits/beauty/out/source-fidelity-r13/inland/comparison.png',
    'comparison_sha256': '621b98dad020c3a8dcf032bc46d569a4c9e885d8e1f0b8bc5b29d868eff5df50',
    'shadow_control_fixture': V2+'fixtures/beauty/source-fidelity-r2/inland-shadow-control/fixture.json',
    'shadow_evidence': V2+'audits/beauty/out/source-fidelity-r13/inland/shadow-evidence.png',
    'shadow_evidence_sha256': 'd00046ade5ae15efbbed33132a5b97775907a56d4e11fe99e5cb2db7bd1124b4',
    'shadow_detail': V2+'audits/beauty/out/source-fidelity-r13/inland/shadow-detail.png',
    'shadow_detail_sha256': '5e75837ac9ba4d139a15f396c28f6f51cd277b3dab2b8bfae4a660f773c011f9',
    'shadow_evidence_record': V2+'audits/beauty/out/source-fidelity-r13/inland/shadow-evidence.json',
    'disposition': 'agent_visual_qa_pass_ready_for_game_integration',
}
