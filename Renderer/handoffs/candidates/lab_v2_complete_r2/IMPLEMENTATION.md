# Implementation map and pickup order

## 0. Begin with the current source-fidelity state of the art

Before using the older composed-system catalog below, read
[LAB_STATE_OF_ART.md](LAB_STATE_OF_ART.md) and consume
[LAB_STATE_OF_ART.json](LAB_STATE_OF_ART.json). The four per-system Metal
witnesses remain the detailed source-of-truth; `source-fidelity-r13/inland` is
the accepted coexistence proof using their exact terrain, mountain and forest
paths with shared source-mesh shadows and retained hydrology. Cities are excluded from this
update: preserve the pre-existing r2 city catalog and implementation without
treating any city experiment as part of this pickup.

Carry each system independently. Preserve the authored mountain relief, tree
bodies/recipe/masks/normals, Warrior geometry/UVs/proportions, Warrior
address modes, terrain material families and high-quality sampling contract.
Only translation, rotation and uniform XYZ scale are permitted for intact
objects unless a later visual audit proves a different need. Missing source
behavior stays explicitly pending; do not substitute procedural crowns, fake
snow decals, guessed LEAN normals or flattened geometry.

Port the r2 viewport adapters around the exact `beauty_terrain` and
`beauty_mountain` providers. Then add the source-body forest recipe, materials,
building/river/coastline exclusions and the selected opacity-aware directional
cast shadows. Do not port the rejected r4 `frozen_l21` approximation or contact disks. The
superseded scene that allows trees inside buildings is not a fallback or an
acceptance witness.

The r13 shadow correction is mandatory, not an optional polish pass:

1. Build one Q6 light-space frame from the captured `EnvironmentState`.
2. Bind its normalized `ShadowL` to terrain, mountain and forest shaders and
   use that exact vector for diffuse/specular face lighting as well as the
   source-triangle shadow projection. Do not retain provider-local sun vectors.
3. Convert every static natural vertex to the same world basis before either
   projection or shadow submission. At the Lab viewport, the check equation is
   `screen_x = 40 + (world_x + world_y) * 64` and
   `screen_y = 380 + (world_x - world_y) * 32 - world_z * 52.48`.
4. Preserve forest caster + receiver + alpha-cutout metadata and sample the
   authored opacity mask; do not substitute circular contact disks.
5. Run the matched shadow control. It must change only the Q6 receive flag and
   must leave city rendering absent/unchanged. The pinned r13 proof shows one
   common noon up-right cast direction and 24,386 meaningfully darkened pixels.

All paths below are repository-relative. Native ownership is based on the
Integration agent’s read-only report at commit
`a0683e0d2a5cb0694961d1993b4c56a8aa1d1323`, independently checked against source.
API 14/pickup-r1 is already integrated, including later caching, world-state,
material-binding, 16-bit copy, input-aware scheduling and animation fixes.
The staged DLL reported SHA-256
`f16d9474faa3466e843f53001ef5350cb4acb38fbfd40f79cc3376bba1be917e`.
Neither installer nor Civ III was run for this consolidation.

## 1. Establish the common frame and preserve native work

Keep `native/environment_runtime.cpp::evaluate_environment`, authoritative
Civ III screen anchors, world capture, stable identities and native action
cursors. Reconcile the posed position/normal/height basis using [SHADOWS.md](SHADOWS.md)
before adding animated shadow receivers. No per-category clock or light
selection. Keep dynamic pose/lighting data out of static terrain cache keys.

Keep the input-aware optional redraw scheduler, 32 MiB resource pose buffers,
24 MiB resource MSAA backdrop cache, 8 MiB/128-entry unit sprite cache, world
shadow-page budget, bounded cancellation and performance instrumentation.
Do not replace `native/c3x_renderer.cpp` with `shared/frozen_scene.cpp`: the
latter is a Lab provider/reference implementation, not a production renderer.
Preserve `c3x_renderer_unit_draw_background`, successful `Unit.Body.Rect`
expansion, RGB555/RGB565 copy behavior and native body fallback for unsupported
actions. These prevent split bodies, stale pixels and magenta edge artifacts.

## 2. Compose the natural scene updates

| System | Selected entry points under `Renderer/terrain_lab/v2/` | Native landing point / constraint |
| --- | --- | --- |
| Terrain, shore, hills, mountains, volcano | `fixtures/beauty/source-fidelity-r2/inland/{terrain,mountain,hydrology}.module.json`, `systems/relief/{beauty_terrain,beauty_mountain}.cpp`, `shaders/relief/{beauty_terrain,beauty_mountain}.hlsl` | Retain `native/profile_v2` capture/cache and optimized world queries. Port the exact selected providers: distinct terrain families, independently seeded authored hills/rock footprints, five authored mountain macro variants and complete 2K material channels. Use the canonical world projection and shared `ShadowL`; the old mountain x-origin of 104 after world normalization is forbidden. Retain hydrology after replacement relief and suppress the old hill/mountain geometry. |
| Forest / canopy | `fixtures/beauty/source-fidelity-r2/inland/forest.module.json`, `systems/objects/beauty_objects.cpp`, `shaders/objects/beauty_objects.hlsl`, `systems/objects/canopy_layout.h`, `systems/lighting/alpha_coverage_v1.h` | Use the BeautyStudies inventory: 22 bodies, 25 recipes, total source count 180, opacity masks, packed normals and source address modes. Preserve readable key/fill face separation, caster + receiver + cutout metadata, the shared `ShadowL`, and authoritative exclusions. |
| Sampling / final reconstruction | `fixtures/beauty/source-fidelity-r2/inland/fixture.json`, `shaders/sampling/linear_reconstruct.hlsl`, `app/runner.py` | Preserve scene-linear output and one final reconstruction at 4x MSAA, 16x anisotropy, 2x render scale and -1 mip bias. Do not downsample or precompress source materials before the renderer owns the final gameplay-size image. |
| River shape and bank placement | `systems/hydrology/{river_corridor,field,scene_adapter}.h`, `qa/river_corridor_pass.py` | Preserve canonical Civ III river edges/topology. Share corridor/exclusion data with terrain, trees, rocks, objects and bridges. Headwater pool, mouth and terrain relief must use the same field. |
| Natural water and reflections | `shaders/hydrology/{water_natural,planar_reflection_pass}.hlsl`, `qa/water_reflection_pass.py` | GPU reflected geometry prepass → water sampling in scene-linear radiance → single final exposure/transfer. Reflection provider halo includes offscreen casters. No pre-tonemapped reflection texture or new presenter. |
| Source shadow receiver correction | `systems/lighting/scene_shadow.cpp`, `systems/lighting/{shadow_field_v1,alpha_coverage_v1}.h`, `shaders/lighting/shadow_visibility_v1.hlsl`, `qa/source_fidelity_shadow_evidence.py` | Keep production source pages/alpha support and texel-derived normal offset; retain the world-space receiver plane from unshifted geometry. Require the matched on/off proof and one face/cast light vector before visual sign-off. |

The natural water pass inherits the newer river/canopy stack in the three main
regions; longcoast and freshwater have explicitly different foundation fixtures.
Use `manifest.cases`, not a presumed identical revision ancestry. The original
test.biq contains zero volcanoes: `combinedvolcano` is synthetic and clearly
separate. Main coastal/inland/wilderness and documented holdouts keep 100-tile
input regions, matched cameras, placement and output sizes. They do not replace
the existing 192-tile/four-phase gate requirements.

Do not turn on later surface-richness diagnostics just because their files are
newer. Complete source-ground layering, mountain height/normal reconstruction,
LEAN variance/environment calibration and unapproved analytic dunes remain
explicit gaps. The current water choice does not authorize surf/animation.

## 3. Preserve cities unchanged in this pickup

Do not port, replace or retune cities as part of the source-fidelity work. The
existing r2 selections and limitations below remain historical Integration
context only. The new natural witness renders no cities. The only new cross-
system requirement is to expose authoritative city/building footprints to the
forest placement stage so `ClipBuildings=true` prevents trees inside buildings.

When city work is separately resumed, retain the existing composed pipeline:

The reference orchestration is `qa/city_central_capital_probe.py` (r111/r112),
with the following dependency order:

1. `qa/city_scene_pass.py` and `systems/objects/presentation.py` resolve generic
   source parts/materials and authoritative era/culture/size/capital input.
2. `city_growth_layout.py`, `city_ground_geometry.py`, `city_exclusion.py`
   and `city_generator_layout.py` preserve uniform source proportions,
   growth prefixes, terrain/river/forest clearance and deterministic layout.
   Use `--central-capital --orthogonal-buildings` for the selected modern cases.
3. Use one composed source transform for bodies, normals/tangents, ground,
   emissive UVs, source-hull paving, lights and shadow geometry. The American
   palace has a baked -30° footprint, so its offline +30° correction precedes
   quarter-turn layout choices. Other houses need no correction in these cases.
4. `qa/city_facade_light_probe.py` derives capital lights from actual facade
   planes. `qa/city_light_buffer_probe.py` uploads the generic bounded arrays.
   Capital lights must not be projected onto the unrotated AABB again.
5. `qa/settlement_ground_probe.py --capital-footprint source-hull` and
   `systems/objects/settlement_ground.py` connect paved ground. The paving
   shader receives the same local irradiance as terrain. Preserve source UVs.
6. `qa/city_environment_probe.py --enable-bound-metalness` enables the selected
   modern material response, followed by GPU planar reflection and HDR glow.
   Do not apply the rejected Asian roof environment trial globally.

Material appearance needs the full channel contract, not diffuse-only meshes:
UV0 diffuse/normal, recovered UV1 AO and UV2 light map where present, decoded
source tangent directions, reconstructed normal Z, roughness/F0, bound metalness,
opacity/cutout and emissive. Check actual per-material presence; missing roles
use explicit generic defaults. The environment fallback is authored analytic
sky/ground, not recovered source cube/SH; LEAN1 variance remains unresolved.
The selected extended city feature payload is 23 floats/92 bytes; legacy
features use 13 floats/52 bytes. Never reinterpret old vertices as new ones.

`systems/objects/capital_styles.json` and the offline
`tools/asset_compiler/palace_asset_importer.py` supply opaque generic roots.
Civ III capital state is authoritative; civilization/style → culture → default
selection is modder-defined. There are 47 standard palace assets; scenario-only
bindings are excluded and four Gran Colombian child trees remain unresolved.
Do not infer capital status from a Civ VI ID or place palaces in ordinary cities.

The selected central modern recipe is proven at inland/freshcanopy only. The
coastal seven-house surrounded fit fails even after 25 legal core attempts.
Keep the explicit coastal fallback and resolve footprint/foundation composition;
do not erase terrain, move forests or silently shrink source proportions.
Asian/ancient selections retain one era but still need the latest central-grid
preference applied. Wall envelopes and all culture × era × size cases are not
certified by these two modern witnesses. Connecting-road design stays deferred.

## 4. Carry forward the older object systems explicitly

| System | Lab authority under `Renderer/` | Integration action |
| --- | --- | --- |
| Roads / rails | `handoffs/L14_roads.json`, `L15_railroads.json`; `terrain_lab/terrain_lab.cpp` | Already integrated. Preserve graph/bridge rules and native cache work; no new road design. |
| Resources | `handoffs/L16_resources.json`; current `native/animation_runtime.*` and animation checkpoint | Static source bodies already cast/receive. Current animated subjects need posed shadow work; keep the newer runtime. |
| Walls / mines | `L17_cities.json`, `L18_mines.json`; city/wall/mine runtime builders | Existing native paths. Compose materials/shadows with the new frame and city footprints. |
| Farms / irrigation / tundra | `L19_farms_tundra.json`; `terrain_lab/fixtures/l19_*.csv` | Code exists in native pickup, but formal I19 is still pending. Retain independent tundra material. |
| Goody huts / colonies | `L19A_goody_huts_colonies.json`; `tools/asset_compiler/{tile_object_asset_importer,build_tile_object_runtime}.py`; `terrain_lab/terrain_lab.cpp::add_tile_object_scene` | Not integrated. Consume viewer-visible authoritative object, owner, era and matching-resource state. Resolve the frozen runtime hash failure before claiming replay acceptance. |
| Fortress, barricade, airfield, outpost, radar, victory, pollution, crater | `L19B_remaining_tile_infrastructure.json`; three runtime bundles listed there; `terrain_lab/terrain_lab.cpp` | Not integrated. Raised objects are casters, flat state art is not. Preserve source compound placement and state-removal semantics. |
| Territory borders | `L21_complete_beauty_scene.json`; `shared/frozen_scene.cpp::{add_territory_scene,add_territory_boundary,add_territory_segment}` | Not integrated. Replace synthetic `build_lab_territory_owners` with authoritative visible per-tile ownership. Emit discontinuities once, suppress same-owner seams; each ribbon uses its owner's main color, rounded joins and restrained waviness. |
| Units / compound bodies / action effects | `L20_units.json`, current `docs/i20_native_unit_animation_handoff.md`, animation checkpoint | Current nine-family/79-action native runtime supersedes the old pose path for supported bodies. Preserve offline compound/member/tool/action contracts; broader family, companion, formation and VFX coverage remains open. |
| Barbarian camps | `docs/barbarian_camp_import.md`, `tools/asset_compiler/tile_object_render_strategy.json` | Separate offline-only conversion. Never substitute a goody hut or imply L19A approval covers camps. |

Borders currently remain native overlays. Before enabling custom territory,
define the exact retained/custom boundary, fog/visibility clipping, zoom/wrap
ownership and invalidation; avoid duplicate native/custom ribbons. Historical
L21 reference hashes are explicitly **pre-territory**; final border matrix was
historically waived, so fresh matched composition evidence is still needed.
Do not treat geometry availability as proof of capture/suppression hooks.

## Binding, depth and ownership contracts

| Interface | Required merge rule |
| --- | --- |
| Lab Q6 frame / native viewport | Lab `b1` contains the Q6 frame and city-light extension. Native `b1` is viewport settings; native shadow settings are `b2`, world settings `b3`, shadow page table `b4`. Allocate/remap a native binding explicitly; do not paste declarations over viewport data. |
| Local city lights | 80-byte frame prefix plus 7,216-byte bounded light/blocker payload; 128 lights / 32 blockers. Preserve prefix and prove byte offsets against `qa/append_frame_data.cpp` and buffer tests. This is Lab ABI, not a mandated native slot. |
| Textures | Native static alpha caster pass uses its own 33-slot binding table; resource material bindings start at `t116`, shared shadow receiver uses `t25`/`t17`. Lab slots and reflection bindings collide with these. Build a per-pass explicit binding map and clear SRV/RTV conflicts. |
| World / normals / depth | See shadow contract. Apply pose then uniform model transform, then canonical basis conversion, inverse-transpose normals and authoritative anchor/depth. Normals are not positions. Paving and source ground use the same transformed footprint. |
| Output | Scene-linear premultiplied MSAA color → reflection/material composition → exposure/glow/transfer once. Do not bloom Civ III overlays, labels, selection, fog or HUD. |
| Coverage | Custom-on m19 remains the complete renderer-owned map plane, fail visibly on invalid ownership; config-off is vanilla. No native-terrain replay fallback. Unsupported unit bodies retain their separate native animator fallback. |
| Dependencies | Static layout/material/environment revisions invalidate bounded object and receiver halos. Animation phase updates dynamic shadows and posed overlays only; scrolling translates/reuses world data. Removal invalidates the union of old and new body/cast/receiver bounds. |

No new patch symbols are requested for this preparation. Existing GOG unit
inleads remain sufficient for the current supported path. If a concrete new
capture/suppression dependency is needed, integration must update the patch
ledger with exact symbol/signature/build addresses and fallback, then use the
existing user checkpoint. Never edit `civ_prog_objects.csv` from this package.
