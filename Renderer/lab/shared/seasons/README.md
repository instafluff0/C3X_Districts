# Seasonal shading

Shader code for C3X's seasons, from the approved Lab looks: Autumn v18
(`lab/studies/autumn`; since 2026-10-26 grassland and plains take two hue shifts,
khaki-olive and tawny, in `autumn.hlsl`), Winter v24-laden (`lab/studies/winter`), and Spring v20
(`lab/studies/spring`). Plan and decisions: `docs/seasons_integration_plan.md`.

- `autumn.hlsl`, `winter_natural.hlsl`, `winter_integrated.hlsl`, and `spring.hlsl` are
  the file-scope helper libraries extracted from the studies.
- `seasons.py` lists every insertion, anchored in the flattened shaders. Each
  season's code runs inside `[branch] if (C3X_SEASON_INDEX == n)`: 1 is Autumn,
  2 Winter, 3 Spring. Summer (0) runs exactly the code it ran before. Where a
  study rewrote a line, the original line stays and the season overrides its
  result inside the guard.
- The season reaches the shaders through `EnvironmentState::season`:
  - integrated shaders read it from `game_resource_look.z` (`TerrainSettings`);
  - relief, mountains and objects read it from the natural `Frame` constants
    (`C3XSeason`, the 7th `float4`).
- Every insertion is marked `/*c3x-season:TAG*/ ... /*c3x-season-end*/`, so
  `strip()` restores the original bytes and `apply()` can be re-run.

Use `tools/season_shading.py`:

- with no option it writes the runtime pack (backup + `season-overlay.json`);
- `--lab-root DIR` writes a Lab shader root for `C3X_RENDERER_SHADER_SOURCE_ROOT`;
- `--strip` removes the seasons.

Re-run it after any other `overlay_*_shading.py` tool, because those tools can
copy a region without the seasons back over the pack.

Game path:

- Renderer64 rewrites some shader text at run time (`sandbox/fresh_pipeline.h`,
  `sandbox/terrain_material_fast.h`). Season code stays outside those ranges,
  and the relief turf edits take their own grass samples.
- Open water in the game is `sandbox/water_surface.hlsl`, which already tints
  Coast, Sea and Ocean. Each season recolours Sea and Ocean relatively there: a
  hue offset and chroma/lightness scales, faded at night.
- The coast keeps Summer's translucent look in every season (user). Shallow beds
  never change, and every water grade fades in from Coast to Sea.
- Jungle keeps its look in Autumn and Spring, like conifers. The natural runtime
  marks jungle bodies in `instance_secondary.z`, and the tree vertex shader
  gives them season seed -1.
- `C3X_RENDERER_SEASON_OVERRIDE=0..3` pins the renderer's season for captures.
  The game's clock is unchanged.

Tests: `tools/test_season_shading.py` (in the day-night category).
