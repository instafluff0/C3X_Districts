# M6.3 Evidence

- `Renderer/default.custom_rendering.txt` is the runtime default layer and uses stable M6.1 logical IDs.
- `Renderer/native/terrain_definition_runtime.cpp` merges default, scenario, and custom terrain definitions without source-specific runtime formats.
- `Renderer/native/native_smoke.cpp` proves multiple texture slots, layered override, deterministic map-coordinate UV sampling, and atomic transparent fallback for a corrupt dependency.
- Live feedback exposed that Civ III's staggered fallback underlay cannot be restored one logical tile at a time. `injected_code.c` now validates the exact fallback indices, then replays the complete native base underlay before opaque accepted diamonds are composited, preventing black wedges beneath retained hills/features without changing their ownership.
- `Renderer/tools/asset_compiler/terrain_pack_builder.py` compiles six proven local material families into the ignored source-agnostic `TerrainNormalized` pack.
- `Renderer/terrain/m6_3_runtime_coverage.json` classifies all 14 terrain types plus transitions, polar ice, and landmarks as mapped or explicit Civ III fallback.
- `m6_3_definition_terrain` is the portable gate; `m6_3_local_terrain_art` rebuilds and renders the locally installed licensed art without committing converted payloads.
- `TEST_INJECTED_CODE_COMPILE.bat` passes. M6.3 uses the existing map capture/insertion symbols and requires no `civ_prog_objects.csv` entry.
