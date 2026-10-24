# Natural Wonder Rendering Contract

## Purpose

Natural wonders are a standalone renderer category. They are not ordinary BIQ landmark terrain, generic mountains/volcanoes, constructed Great/Small Wonders, or ordinary C3X districts. Current C3X represents each placed natural wonder as a completed `NATURAL_WONDER_DISTRICT_ID` instance with an authoritative natural-wonder ID and tile coordinate. The renderer observes that state and replaces only its visual body.

Natural wonders are M9. Constructed wonders move to M10, and districts remain the final milestone as M11.

## Authoritative C3X Inventory

The inventory must parse the effective natural-wonder definition set using the same replacement precedence as C3X:

1. Scenario `scenario.districts_natural_wonders_config.txt`, when present.
2. User `user.districts_natural_wonders_config.txt`, when present and no scenario replacement applies.
3. `default.districts_natural_wonders_config.txt` otherwise.

The checked-in default currently supplies 18 `#Wonder` blocks, but that is only a mapping seed. C3X supports up to 64 definitions, dynamic replacement sets, and scenario-authored identities; tests must not hard-code 18 as complete runtime coverage.

Each effective definition and placed instance records:

- Stable natural-wonder ID/name and config provenance.
- Authoritative tile coordinate and pixel anchor from the C3X district instance.
- Required base terrain, optional adjacent terrain/river requirement, and adjacency direction.
- Native `img_path`, row, column, and any selected proxy/animation metadata needed for exact fallback and diagnostics.
- Up to eight configured animation alternatives, including hour mask, season mask, direction, frame time, and pixel offsets.
- Visibility/exploration/fog state and whether the retained native name label is eligible to draw.
- Visual orientation, bounds, depth envelope, environment state, and deterministic variant seed.

BIQ landmark terrain and volcano tiles remain M6 terrain selectors unless an actual C3X natural-wonder instance identifies the tile. A similar name or appearance never promotes a terrain tile into this category.

## Source Mapping

The Civ VI adapter inventories Base and DLC `Features.artdef`, the `NaturalWonders` collection in `TerrainStyle.artdef`, `Clutter.artdef`, `VFX.artdef`, `Water.artdef`, `WaterMaterials.artdef`, `Wave.artdef`, `Landmarks.artdef`, and referenced cooked packages. It maps C3X names to source candidates with `exact`, `approximate`, or `authored_required` status, separately reports unavailable installed targets, and retains evidence for model pieces, terrain materials, clutter, water integration, orientation, VFX, emissives, and lights.

The checked-in first pass is `Renderer/inventory/vanilla_c3x_to_civ6_natural_wonders.json`, documented in `Renderer/docs/civ3_to_civ6_natural_wonder_mapping.md` and enforced by `Renderer/inventory/natural_wonder_mapping_inventory.py`. It covers every default PCX cell and animation count, distinguishes exact matches from provisional stand-ins and authored requirements, and resolves preferred targets against installed `Features.artdef` definitions. It must remain editable conversion input rather than becoming a runtime identity table.

Civ VI multi-tile layouts are source-authoring information, not C3X gameplay state. A compiled natural-wonder kit may contain multiple visual pieces and may overhang neighboring tiles, but it is rooted at the single authoritative C3X instance anchor. It cannot create hidden gameplay tiles, suppress neighboring terrain/objects, change adjacency, or extend yields/passability. A future C3X multi-tile gameplay feature would require a new versioned scene contract.

The runtime pack contains only generic assets such as:

```text
natural_wonder/yosemite/base
natural_wonder/yosemite/terrain_blend
natural_wonder/yosemite/clutter
natural_wonder/angel_falls/waterfall
natural_wonder/yellowstone/steam_attachments
```

Runtime code never branches on Civ VI names, ArtDefs, BLPs, or source footprint conventions.

## Composition And Ownership

A natural-wonder kit can combine:

- Main mesh or terrain-integrated geometry.
- Base-terrain blend/decal and optional local material override.
- Directional or adjacency-specific pieces.
- Clutter/vegetation/scatter that belongs to the wonder.
- Water surfaces, waterfall sheets/particles, foam, spray, or wave response.
- Static emissive materials and analytic lights.
- Attached smoke, steam, fire, birds, mist, sparks, or other ambient effects.

M6 retains ownership of the underlying terrain and M7 retains its independently resolved infrastructure/resources unless the complete natural-wonder contract explicitly masks a visual subregion. Civ III retains fog/shroud, borders, grid, selection, cursor, minimap, HUD, and UI. The C3X natural-wonder name label becomes renderer-drawn map text (user, 2026-10-23; see Game Integration). C3X still decides what it says and where. The renderer draws it in Civ III's font look, anchored to the tile, unlit above the replacement and under the fog and unexplored overlay.

Replacement is atomic per instance. C3X suppresses the native natural-wonder sprite and its legacy custom animation only after the complete required kit is loaded and the renderer accepts the instance. Missing geometry, required directional pieces, device failure, or unsupported source state restores the complete native body exactly once; renderer-owned attachments disappear with it.

## Environment And Animation

Natural wonders consume M6.4's shared sun/moon, water, emissive, analytic-light, and ambient-attachment system. Waterfalls, geysers, steam, volcanic glow/smoke, waves, mist, and similar motion use M5.3 absolute presentation time and stable instance/effect IDs. Static scenes request no continuous redraw. Hidden, fogged, off-screen, paused, daylight-disabled, fallback, and completed one-shot effects contribute no active frame request.

C3X remains authoritative for hour and season. The renderer may vary snow, vegetation, water, emissives, and effect intensity without changing terrain identity, placement, visibility, yields, movement, or any gameplay rule.

## Verification Gates

### M9.1 Inventory And Mapping

- Parse default, user, and scenario replacement behavior with synthetic fixtures and report every effective definition and placed instance.
- Account for every field, native PCX cell, animation alternative, direction, terrain/adjacency condition, and retained label/fog responsibility.
- Generate a swappable C3X-to-source mapping ledger with confidence, Base/DLC provenance, source parts, footprint/orientation evidence, and unresolved dependencies.
- Prove that ordinary landmark terrain and volcanoes are not misclassified.

### M9.2 Rendering And Integration

- Render deterministic land, mountain, volcanic/geothermal, coastal, and water examples at two viewport sizes across noon, sunset, midnight, sunrise, and representative seasons.
- Cover static, animated, directional, adjacency-sensitive, source-multipart, fog-hidden, scrolling/wrapping, clipping/depth, missing-asset, reset, and config-off cases.
- Prove animated effects remain deterministic across skipped frames and static natural wonders become idle.
- Prove the native name label and all retained Civ III layers remain visible exactly once and are not relit or bloomed.
- Require every effective custom-on natural-wonder instance to resolve to a complete 3D kit or fail visibly without native body replay; partial replacement cannot pass.

Manual screenshots remain a single batched strategic checkpoint after automated scene, contact-sheet, image, ownership, and performance gates pass.

## Game Integration (2026-10-23)

The user approved all 18 Lab kits in all four seasons (`Renderer/lab/studies/natural_wonders/`, ledger `Renderer/inventory/vanilla_c3x_to_civ6_natural_wonders.json`). The Autumn session leads integration and reviews each stage. Natural wonders are a C3X feature, so their authoritative state is C3X's own (`injected_code.c`, `C3X.h`):

- `district_tile_map` instances with `district_id == NATURAL_WONDER_DISTRICT_ID`;
- `natural_wonder_info.natural_wonder_id`;
- `natural_wonder_configs[id]`.

No `civ_prog_objects.csv` entry is needed.

Today Renderer64 owns the whole m19 plane, so `draw_district_on_tile` (inside m12) never draws the wonder sprite. A wonder tile shows only its host terrain. The name label is C3X's `draw_map_tile_text` call in `patch_Main_Screen_Form_draw_city_hud`, captured as world-anchored native ink like city labels.

### Facts (frame ABI 23, after Day-night's 22)

C3X is much more than Renderer64, so injected code reports facts and keeps no renderer logic (the user, 2026-10-23). Its whole change:

- **Tile record:** `natural_wonder[32]`, the effective config name, and `natural_wonder_flags` bit 0 = C3X's `show_natural_wonder_name_on_map`. Records arrive for explored tiles only, through world pages and frames, as the capture already withholds content.
- **Native label:** `draw_map_tile_text` is skipped while the renderer DLL is loaded. C3X keeps the module only when its API version matches, so a loaded DLL draws the names.

No C3X.h state, topology change or `civ_prog_objects.csv` entry is needed.

Renderer64 does the rest:

- **Kits:** names resolve to kits through the pack's aliases (case-insensitive, like resource compositions). Runtime code never branches on a wonder name. A name without a kit leaves the host terrain and draws no 2D art (`zero-native-2D-art`).
- **Relief slots:** relief compiles from the topology word alone, and its validity proofs recheck those words.
  - `ViewerTopology` folds each explored wonder record's relief slot into bits 28–31 of the renderer's viewer topology. A slot is 1–15: the pack's kits with a stamp or foam collars, in pack order (13 today, capped at 15 by the compiler).
  - The viewer revision then invalidates every neighbouring relief proof, exactly as exploration masking does.
  - A wonder's stamp, water and collars therefore appear when its tile is explored, recompiling a few neighbouring tiles once. Before that, the overlay hides the tile anyway.
- **Host terrain:** a stamped wonder sits on its real C3X host terrain (the user, 2026-10-23). The Lab built the kits as a mountain over the C3X ground. On a host like Yellowstone's (grassland beside forest), the forest edge and seating differ slightly from the approved frames; hosted like the Lab, production matches them.
- **Transport impact:** API 23, journal protocol, IPC wire version, the codec field list, `world_preparation.h`'s tail static_assert, and the field-coverage, round-trip and bridge-signature tests.

### Runtime pack `Renderer/packs/NaturalWonderRuntime`

`Renderer/tools/asset_compiler/build_natural_wonder_pack.py` builds it from the approved study. It is generic: kit keys, aliases, fields and meshes, with no Civ VI identifiers.

- `wonders.json`: aliases (config name to kit) and, per kit:
  - an optional terrain stamp: span in tiles, height in relief units, turn, region kind (1 snow, 2 travertine, 3 banded sandstone, 4 stratovolcano), and snow;
  - effect records: ten floats each, offsets from the tile centre. Kinds are geyser, waterfall ribbon, spray, lake and channel;
  - body compositions, each with a season mask.
- `stamps/<kit>_{height,ground,special}.dds`: R8 fields.
- `waves/<kit>.bin`: foam-collar ribbon points, natural offsets from the tile centre.
- `resource_runtime.bin` + `textures/`: the kit bodies, in the composition bundle format with its own 8 texture slots.
  - The 18 kits need about 254 unique meshes (summer plus winter), against the bundle's 256-asset limit. The compiler merges co-placed parts that share an atlas.
  - Season variants are separate compositions. Autumn and spring crowns are recoloured copies of the kit's broadleaf textures; winter pines use the snowy pine meshes.
  - **Slot 7 is the veil slot:** the feature shader blends any resource body sampling it through its alpha (`resource_veil_weight`, `VEIL_SLOT`). The compiler refuses any wonder atlas there, in any season. If a kit ever wants true translucency (spray, mist), slot 7 is the deliberate way to get it.

### Renderer

- **Stamps:** `NaturalData` holds every kit's stamp, effects and waves. `MountainShape::add_wonder` takes the tile's kit. Topology bit 28 comes from the tile fact (the Lab's BIQ marker is retired), and a stamped kit's tile rides the mountain relief path as in the Lab.
- **Region kind:** a runtime per-vertex value replaces the Lab's compile-time `WONDER_REGION_KIND`. It is `volcano_owner.w` where no volcano owns the relief, constant across the stamp's reach; the region weights fade to zero before the reach ends.
- **Water:** the lake and fjord channel join the production bed, water and river layers, and the relief clips at the waterline.
  - Their chunks take their layer's projection: kind 3 in world-ground mode, like every per-tile ground chunk.
  - Their vertices mark `shape_visibility.x` 2, which no other water writes. The shared receiver query (`lab/shared/shaders/lighting/scene_shadow_v1.hlsl`) keeps a third of the cast shadow for marked water. Without that, a fjord channel between tall walls drew as a dark hole in game, unlike the approved Lab look (user, 2026-10-23).
  - The pack gets it through `overlay_natural_wonder_shading.py` (region `wonder_water_shadow`).
- **Foam collars:** collars join the coastal waves.
- **Effects:** geyser, waterfall and spray are city effect layer kinds in `city_scene_material.hlsl`.
- **Bodies:** a wonder body bundle joins the object bundles, like sites and mines. Body chunks carry a season mask, generalising `season_stand` (today 1 = outside Winter, 2 = Winter only) to "shown in these seasons". A season flip changes what draws without recompiling.
- **Seasons:** wonder winter edits of shared shader code are tagged, guarded insertions in `Renderer/lab/shared/seasons/seasons.py`:
  - `winter/wonder-basin` keeps geothermal ground bare;
  - `winter/wonder-waterline` keeps snow off anything at or below the water plane.
  
  Wonder-only branches may read `C3X_SEASON_INDEX` directly.

### Labels

The user wants Renderer64 to draw natural-wonder names itself, for performance (2026-10-23). The injected bitmap option was dropped.

- **Facts:** C3X reports each wonder's id, tile and name, and skips its native `draw_map_tile_text` call. Config-off and declined paths call the native draw unchanged. No csv hook is needed (C3X state).
- **Text:** the renderer draws the name at screen resolution in Civ III's font look, with fixed pixel size at every zoom.
  - The font is `get_font (10, FSF_NONE)`: Lucida Sans, lfHeight −10, normal weight, OUT_TT_ONLY_PRECIS, DEFAULT_QUALITY (font facts: `docs/performance_review_20261007.md` section 49).
  - It is white with a black shadow offset 1,1.
- **Placement:** the same as `draw_map_tile_text` at the presented zoom.
  - The text is centred in the tile width minus a sixteenth.
  - Its top sits a thirty-second of a tile below the 88/128-high wonder cell, raised by the cell's 24/128 overhang.
  - The text box starts 10 px above that top line.
- **Fog order:** in C3X and Civ III, fog and unexplored are drawn last, so a name never shows until its tile is explored. Renderer64 keeps that order: wonder labels draw with the world, under the fog and unexplored overlay, and fog darkens a fogged tile's label like everything else.
- **Where in Renderer64:**
  - **Pass:** `c3x_renderer64_render_fresh` (`sandbox/resident_scene.cpp`). The label pass goes after the tone map and the city-site overlay and before `visibility_gpu.apply` (fog and unexplored), on the same display target, modelled on `GpuCitySiteOverlay`.
  - **Anchors:** each wonder tile record in `frame.tiles` (wrapped copies carry their own) through `SceneProjection(...).x/y`, snapped to whole pixels. The quads keep a fixed pixel size at every zoom.
  - **Text pixels:** one premultiplied texture per label, built once with GDI from the renderer's own memory DC (`native_text_raster.h` can compile from a hand-filled font state).
  - **Ink:** captured HUD ink composites above fog, so the native `custom_renderer_hud_scope` draw must stop for these labels.

### Stages

Each stage goes to the integration lead as a diff, with tests confirmed to fail without the change.

1. Pack compiler and its tests.
2. Seasons insertions (`tools/test_season_shading.py` stays green).
3. Renderer port with Lab parity renders, after ABI 22 lands.
4. Injected facts, labels and config-off delegation tests, with `TEST_INJECTED_CODE_COMPILE.bat`.
5. Renderer64 build, stage, and the scripted game test on a disposable save with natural wonders.

### Verifying in game

Lab renders, including `prod_parity.py` with `--pack-shaders` and `--save`, run the 32-bit DLL, which is built without `C3X_RENDERER64_FRESH`. They never exercise the game's resident world: canonical world content, world objects, shared ground and object components, and world-preparation jobs. A change there can pass every Lab render and still be wrong in game. Verify it with a scripted game run.

On 2026-10-23, the bodies matched the Lab everywhere, but in game they lost their low parts under the ground while their shadows still drew. Body chunks must take the feature layer's projection (kind 2 with world objects), so that `resource_natural_depth` gives them the terrain's height-depth basis. `test_natural_wonder_draw_pairing.py` holds that rule.

For a survey, run `run_scripted_game_test.ps1 -Scenario debug-scroll -Seconds 190 -Route ...`, with camera stops at each placed wonder at 2x and 1x (`tools/scripted_game_test.md`), on the disposable new-game save with `enable_natural_wonders` set temporarily. Debug-reveal captures lay a pale wash over many tiles, and a tall object picks it up from the tile behind it. That wash is a capture artifact, not wonder shading.

## Civ III Patch Dependencies

Start with existing C3X `natural_wonder_configs`, `district_tile_map`, `natural_wonder_info.natural_wonder_id`, `draw_district_for_tile`, the natural-wonder image/animation paths, and the existing retained-label draw hook. Extend the existing visible-scene capture and suppression logic if those fields are not yet exported.

No new Civ III function or `civ_prog_objects.csv` entry is currently known to be required. Audit a dedicated natural-wonder mutation/placement hook only if normal C3X map invalidation cannot expose create/load/replace transitions promptly; do not request an address speculatively.
