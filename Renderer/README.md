# C3X Renderer Workspace

This folder is the standalone home for custom renderer experiments. Runtime C3X should consume generic C3X asset packs; source-specific importers such as Civilization VI tooling belong here, not in injected game code.

## Project Navigation

- `MASTER_PLAN.md`: stable goals, architecture, milestones, gates, and agent handoff protocol.
- `ROADMAP.md`: current human-readable progress and ordered backlog.
- `project_status.json`: canonical machine-readable next step.
- `VERIFICATION.md`: executable completion-gate policy.
- `docs/renderer_workstreams.md`: Renderer Lab/Game Integration ownership,
  per-system promotion, and streamlined Windows VM workflow.
- `docs/renderer_config_spec.md`: proposed scenario-aware asset mapping and configuration contract.
- `docs/environment_lighting_and_ambient_effects.md`: shared sun/moon, emissive, water-lighting, and ambient-effect contract.
- `docs/civ6_lighting_findings.md`: installed Civ VI light/VFX source locations, evidence levels, and extraction backlog.
- `docs/civ3_to_civ6_resource_mapping.md`: editable map-only resource mapping seed and retained native-icon policy.
- `docs/natural_wonder_rendering.md`: M9 C3X natural-wonder inventory, Civ VI mapping, composition, animation, ownership, and hard-failure contract.
- `docs/wonder_and_district_rendering.md`: M10 constructed-wonder and final M11 composite-district contract.
- `samples/config/default.custom_rendering.txt`: concrete future parser fixture.

An agent resuming this project should validate the handoff state before doing work:

```text
python3 Renderer/tools/renderer_dev.py state
```

Use `renderer_dev.py lab` or `renderer_dev.py integration` for routine work and
`renderer_dev.py full` when closing a step. On macOS those Windows-only workflows
dispatch automatically to the running Parallels VM named `Windows 11` through
the `Y:` shared home folder.

There is one canonical Git working tree. In Windows, the familiar installed
`C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete\Conquests\C3X_Districts`
folder links through `\\Mac\Home\...` to the same checkout exposed to automation
as `Y:`, so normal `git add`, `git commit`, `git push`, and interactive
`INSTALL.bat` commands can all run there without copying `Renderer/` or injected
sources between repositories. Civ VI source assets stay on macOS and remain
available directly to Mac tools or through the VM's `Z:` share; only normalized
outputs belong in the runtime pack path.

## Current Slice

1. `tools/asset_compiler/c3x_asset_compiler.py` inventories a local Civ VI install and creates a source-agnostic prototype pack manifest.
2. `preview/render_iso.py` renders a Civ III-style isometric terrain preview from that pack without requiring Direct3D, Blender, or Civ III.
3. `third_party/` contains pinned local references to Civ VI conversion tools.

## Commands

Generate the initial layered vanilla Civ III map-art census (M6.0 remains incomplete until BIQ, atlas-layout, and runtime-index evidence resolves every reported gap):

```powershell
py Renderer\inventory\civ3_art_inventory.py `
  --install-root "C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete" `
  --output Renderer\inventory\generated\vanilla_conquests_art.json `
  --markdown Renderer\inventory\generated\vanilla_conquests_art.md
```

Validate the editable vanilla Conquests to Civ VI unit mapping seed, optionally resolving every selected target against the installed ArtDefs:

```powershell
py Renderer\inventory\unit_mapping_inventory.py
py Renderer\inventory\unit_mapping_inventory.py `
  --civ6-assets-root "Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets" `
  --require-all-targets
```

Validate the editable map-resource mapping seed and resolve its targets against installed Base/DLC ArtDefs:

```powershell
py Renderer\inventory\resource_mapping_inventory.py
py Renderer\inventory\resource_mapping_inventory.py `
  --civ6-assets-root "Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets" `
  --require-all-targets
```

From the C3X root:

```powershell
py Renderer\tools\asset_compiler\c3x_asset_compiler.py discover
py Renderer\tools\asset_compiler\c3x_asset_compiler.py build-prototype
py Renderer\tools\asset_compiler\c3x_asset_compiler.py import-loose
py Renderer\tools\asset_compiler\c3x_asset_compiler.py build-grassland-poc
py Renderer\tools\asset_compiler\c3x_asset_compiler.py extract-civbig <source> <output.dds> --png
py Renderer\tools\asset_compiler\civblp_probe.py
py Renderer\tools\asset_compiler\civblp_material_resolver.py
py Renderer\tools\asset_compiler\terrain_geometry_resolver.py
py Renderer\tools\asset_compiler\grassland_pack_builder.py
py -m Renderer.tools.asset_compiler.terrain_pack_builder
py Renderer\tools\asset_compiler\civ6_lighting_probe.py --assets-root "Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets"
py Renderer\definitions\definition_parser.py --default Renderer\samples\config\default.custom_rendering.txt --mod-root . --scenario-root . --output Renderer\verification\default_renderer_catalog.json
py -m Renderer.standalone.whole_viewport_renderer --scene Renderer\samples\scenes\grassland_viewport.scene.json --default Renderer\samples\config\default.custom_rendering.txt --mod-root . --output Renderer\preview\out\grassland_scene_640x480.png
py Renderer\tools\render_fixture_matrix.py --scene Renderer\samples\scenes\grassland_viewport.scene.json --default Renderer\samples\config\default.custom_rendering.txt --mod-root . --references Renderer\samples\validation\reference_metadata.json --output Renderer\validation\grassland_viewport
cmd /c Renderer\native\BUILD.bat
py Renderer\preview\render_iso.py --pack Renderer\packs\Civ6Prototype\manifest.json --output Renderer\preview\out\civ6prototype_1024.bmp --width 1024 --height 768 --grid 16
py Renderer\preview\render_iso.py --pack Renderer\packs\Civ6Prototype\manifest.json --output Renderer\preview\out\civ6prototype_640.bmp --width 640 --height 480 --grid 8
py Renderer\preview\render_iso.py --pack Renderer\packs\Civ6GrasslandPOC\manifest.json --output Renderer\preview\out\grassland_poc_1024.bmp --width 1024 --height 768 --grid 16 --force-terrain grassland
py -m unittest discover -s Renderer -p "test_*.py"
```

On the documented development machine, require the installed Civ VI integration checks as well:

```powershell
py Renderer\tools\verify_project.py --require-local-assets
```

The prototype pack intentionally uses generated placeholder preview materials until a safe loose-source extraction path is confirmed. It records Civ VI source candidates and diagnostics without copying Firaxis assets into tracked files.

`import-loose` reads `Renderer/samples/ManualLooseSource/source_manifest.json` by default and emits `Renderer/packs/ManualLoosePrototype/manifest.json`. This is the source-agnostic path modders and future Civ VI/Civ VII importers should target.

`build-grassland-poc` is the first Civ VI-to-Civ III semantic bridge. It follows Civ VI `TerrainStyle.artdef` grassland entries into the cooked terrain material package, extracts the standalone `TEXTURE_TER_Grass_Decal_B` `CIVBIG` payload to DDS/PNG, and emits `civ3_tile_art_map.json` rules keyed by Civ III `SquareType` plus optional `SquareParts` sheet/sprite metadata.

`extract-civbig` handles the standalone 48-byte `CIVBIG` texture wrapper used in `BLPs/SHARED_DATA`. It validates the complete block-compressed mip chain and writes a standard DDS with a DX10 header. `--png` uses the pinned DirectXTex `texconv.exe` from CivNexus6.

`civblp_probe.py` reads only the 28-byte `CIVBLP` file header and its declared package-data region. It follows typed allocation pointers from `ART_DEF_TERRAIN_MATERIAL_GRASSLAND` to candidate texture records and writes `docs/civ6_grassland_material_probe.json`; it deliberately skips the large embedded payload and does not claim generic package decoding or texture-role semantics.

`civblp_material_resolver.py` uses cross-record type/class consistency to assign the base-color, height, specular, and FOW roles, preserves the consistently null fuzz slot as inferred, and validates each texture's embedded offset and exact BC mip-chain size. It writes `docs/civ6_grassland_material_binding.json` without reading or copying the payloads. The resolved grassland textures are embedded in the `.blp`; matching standalone `SHARED_DATA` files are not required.

`terrain_geometry_resolver.py` follows `TERRAIN_GRASS` through the explicit `Flat` terrain type and `StandardFlat` grassland-material binding. It inventories the geometry-bearing cooked packages without reading their payloads, separates the authored continental-hill relief reference from the flat base, and emits a generic unit-grid mesh with validated topology, normals, and UV0 plus `docs/civ6_grassland_geometry_uv.json`.

`grassland_pack_builder.py` completes the local grassland slice in one command. It extracts the M1.3-validated embedded base color to standard DDS, combines it with the normalized flat mesh, validates that runtime JSON has no source-specific formats or absolute paths, and writes deterministic 640x480 and 1024x768 textured PNG previews. The generated pack, DDS, build report, and images are ignored local artifacts and must not be redistributed.

`terrain_pack_builder.py` expands the local native pack to the six flat-surface families with complete proven dependencies: desert, plains, grassland, tundra, coast, and sea. `default.custom_rendering.txt` maps those stable logical IDs while the tracked M6.3 coverage ledger explicitly leaves incomplete relief, feature, ocean, ice, and landmark families with Civ III. Converted payloads and source evidence remain ignored local artifacts.

`native/environment_runtime.cpp` is the M6.4 shared day/night foundation. It derives continuous sun, moon, ambient, exposure, shadow, emissive activation, and bounded water response from C3X's captured hour/season, and defines generic analytic-light and ambient-attachment records. The synthetic material/attachment fixture proves static emissive idle, absolute-time animated phase, visibility/state gating, and explicit fallback while all non-terrain Civ III layers remain retained.

M6.5 and M6.6 are historical terrain-production gates. M6.7 established the game boundary; paired I9-I11 now consume the frozen, visually approved terrain, dune, and marsh handoffs with exact occurrence ownership, bounded cache/invalidation, both zooms, clipping, scrolling, wrapping, and reset recovery. Custom-on owns the complete `m19` map plane and never replays native terrain after failure, so unported categories are intentionally absent instead of silently mixed with Civ III art. Renderer Lab remains the sole owner of visual changes; every future L# feeds the same-numbered I#.

`preview/render_textured_patch.py` is the matching source-agnostic preview path. It reads the generic pack, mesh, material, and BC3 DDS files, rasterizes interpolated UV0 through the configured isometric projection, and writes a dependency-free PNG.

`definitions/definition_parser.py` and `definitions/rule_resolver.py` implement the v0 renderer-definition contract. The parser produces typed sections with structured file/line diagnostics, merges the `default -> scenario -> custom` layers by complete-section replacement, preserves disable tombstones, validates references, and prevents pack or asset paths from escaping their configured roots. The pure resolver applies the documented selector vocabulary and exact four-stage ranking, emits complete winner/loser explanations, computes deterministic coordinate variant seeds, and preserves fallback without loading asset payloads.

`scenes/scene_contract.py` implements the v0 replay boundary. It strictly validates source-independent viewport, pixel projection, environment, tile, and object records; verifies deterministic IDs, seeds, coordinates, and anchors; produces byte-stable canonical JSON; and can replay a recorded fixture through the rule resolver without Civ III. See `docs/visible_scene_contract.md` and `samples/scenes/grassland_viewport.scene.json`.

For M5.2, enabling custom rendering automatically exports the first successfully composited frame to `Renderer/validation/live/civ3-live.scene.json`; `Ctrl+Shift+F12` optionally recaptures it. Users only need to report visible breakage or odd behavior. Strict schema, category, determinism, and offline-batch checks remain automated; see `docs/m5_2_offline_pipeline.md`.

M5.3 adds the pure native absolute-time scheduler in `native/frame_scheduler.cpp` and connects it to Civ III's existing 66 ms UI timer. The timer hook can only mark Civ III's animator dirty; normal map draws remain the sole capture/render/blit entry. Static scenes request no frames, one pending bit prevents queue growth, and bounded QPC telemetry covers capture, native render/readback, blit, and total map-pass time. No animated category is enabled until its later ownership gate passes; see `docs/runtime_animation_and_frame_pacing.md`.

`standalone/whole_viewport_renderer.py` is the M4.1 whole-scene reference renderer. It resolves the recorded scene through the merged catalog and normalized packs, projects meshes from authoritative Civ III anchors, depth-tests the complete viewport, applies hour/season lighting, clips to the map rectangle, and supports explicit target recreation and teardown. See `docs/standalone_renderer.md`.

`tools/render_fixture_matrix.py` is the M4.2 visual-validation harness. It renders two viewport sizes across four hours and four seasons, writes deterministic structural metrics and dependency hashes, checks required environment differences, and assembles a labeled contact sheet. Cross-engine references remain qualitative and explicitly separate from exact C3X regression hashes. See `docs/fixture_matrix.md`.

`native/c3x_renderer.cpp` is the native implementation boundary, with layered terrain selection isolated in `native/terrain_definition_runtime.cpp`. `native/BUILD.bat` builds its required 32-bit `Renderer/bin/C3XRenderer.dll` and runs an executable smoke test. The DLL uses D3D11 only for an off-screen BGRA render target, normalized BC3 terrain sampling, and staging readback; it creates no window or swap chain. Injected C loads the versioned exports only when `enable_custom_rendering = true`, supplies default/scenario/custom definition paths, captures Civ III anchors during the first `m19` pass, and composites at the audited pre-`0x1fef0` boundary. The current diagnostic omits the original `m19` calls afterward, so unported tile layers do not appear.
