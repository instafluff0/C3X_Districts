# Production pickup integration

The user explicitly requested this port on 2026-09-06, superseding the earlier
pause awaiting production logs. **Ready for the user-run gameplay checkpoint.** The full workflow and launcher
dry run pass; the API 14 DLL is staged. The user launches Civ III. No Lab visual approval, LQ completion, or M9–M11 work is claimed.

## Pinned input

Source: [C3X implementation pickup package](../../handoffs/candidates/lab_v2_terrain_lighting_r1/README.md).
`reference/` is extracted from its immutable archive; `provenance.json` records
the exact closure. Active Lab source changes are outside this port.
`verify_pickup.py` verifies all 358 archived source members, 233 local art
dependencies, baseline handoffs, eight reports and 32 retained images. Unlike the
package's original verification, it does not require evolving live Lab sources
to match the archive. Historical handoffs remain unchanged.

## Implemented

- Continuous material/normal fields, profile-2 coast, selected direct hill
  height, 1.30 mountain and 1.60 volcano relief, source-owner volcano footprint
  and authoritative owner activity. Existing vegetation placement is retained;
  only grounding follows the new surface. Routes, resources, cities, mines and
  farms retain their production ownership and placement contracts.
- Six selected source cliffs and all base/normal/height/gloss channels, promoted
  into the local generic `TerrainProfileR1` pack. The adapter preserves mesh and
  BC payload bytes; gloss DDS headers request the pinned linear view.
- Actual opaque/cutout source casters, shared sun/moon lighting, receiver offset,
  contact and comparison filtering. RGBA16F MSAA4 composition resolves once
  before exposure, shoulder and sRGB conversion to native straight-alpha BGRA.
- API 14 immutable world topology: four bytes per playable tile, refreshed from
  existing Civ III symbols. Sparse indexed coast dependencies and local packed
  tile dependencies invalidate geometry after authoritative edits. Capture uses
  an eight-raw-tile appearance halo and twelve-raw-tile topology halo; only the
  visible scene and necessary source-caster ring block foreground rendering.
- World-keyed mesh reuse, retained recent viewports and prepared pixel blocks.
  Unchanged unit/UI redraws return the published bitmap without cancelling
  background preparation, while reporting current native animation demand.
  Direction-prioritized preparation covers Civ III's multi-tile scroll jumps.
- Full-precision 48-byte feature vertices and shared flat-layer GPU buffers keep
  mesh storage within the existing 192 MiB limit. Existing 64 MiB prefetch,
  16 MiB pixel-block and 128 MiB retained-viewport limits are unchanged.
- Complete-source/compiler-keyed shader bytecode caching with payload checksum,
  bounded reads and atomic replacement removes repeat shader compilation.
- Extensive `OutputDebugStringA`/optional trace-file records cover capture cost,
  topology edits, cache paths, mesh phases, source-shadow reuse, uploads,
  readback, preparation, cancellation, reset and failure. No per-vertex logging.

Production adaptations are explicit: cliff exclusion uses canonical world-cell
greedy order rather than fixture-crop order. Shadows use a separate bounded
128 MiB atlas of 32 world-aligned six-tile pages at 1024 square, with R32 physical
light depth instead of replicated normalized R16. Rendering uses 512-pixel
scissor batches to respect this bound at larger viewports. These adaptations
still require the final combined-scene and pixel-parity evidence.

The inherited analytic dune proxy remains unapproved source recovery. Volcano
BC5 channel semantics remain unresolved; the rejected reconstruction is excluded.
Concurrent Lab rivers, canopy and surface-richness experiments are not included.

## Verification

[checkpoint.json](checkpoint.json) records measurements and local report paths.
The pickup replay passes cached/cold parity: three pixels beyond the rounding
allowance, total channel error 33 across 2,457,600 bytes. Terrain edits rebuild
84 meshes/reuse 304; activity edits rebuild 24/reuse 363. Both edited images
match a cold reset exactly. Existing thresholds are unchanged.

A 960×640 replay with 387 foreground/caster tiles completed prepared fixed jumps
in **18–39 ms**, with **zero mesh builds or uploads**. Repeated unit-style redraw
p95 was **1.30 ms**, maximum 2.14 ms. Preparation completed without cancellation
or unavailable meshes. Initial preparation used about 60 MB; observed total
mesh storage stayed below 164 MB. Cold geometry still takes roughly 10–13 seconds.
These are off-screen measurements, not proof of vanilla-speed gameplay.

All 18 matrix cases pass: both zooms, four lighting phases, horizontal wrap,
synthetic city/walls/capital, resource, mine/farm, road/railroad and active-volcano
witnesses. Image inspection shows continuous coasts, grounded source bodies,
source cliffs and retained infrastructure. This is engineering review rather
than new Lab visual approval. The original BIQ is unchanged.

Final Windows hardware checks pass four SM5 shader entries, real source cutout
holes, warm page reuse, local/distant caster invalidation, MSAA4 and linear
transfer. The frozen renderer's clipping mismatch is resolved by repainting a
32-pixel collar around current/translated viewport clipping boundaries: its
jump comparison improved to 174 pixels / 6,657 error, with prepared-block parity
547 pixels / 19,226 error. The pickup MSAA path does not use this extra collar.
The frozen shader file remains byte-identical to its historical handoff.

The workflow rejects a printed native `FAIL` even if Parallels reports exit zero;
only an empty transport error can retry. Actual test failures are not retried
until a corrective change is made. The temporary low-disk VM interruption has
been resolved. The full workflow passed 81 unit tests, native regression, both
legacy previews, approved injected compilation and the live-checkout link. The
launcher dry run passed; the staged DLL is byte-identical to the tested candidate.

## Reproduce and test

Use `renderer_dev.windows_command_result` for `call BUILD.bat candidate-only` in
`Renderer/native`; it compiles/runs smoke checks without replacing `Renderer/bin`.
`candidate-compile` compiles only. `renderer_dev.py full` includes the integration
unit suite, native regression, both legacy preview zooms, approved injected smoke
and live-checkout verification, and stages the DLL. `verify_native.py` never
installs or starts Civ III. Local images, CSVs and traces are ignored outputs.

```sh
python3 Renderer/native/profile_v2/verify_pickup.py
node Renderer/native/profile_v2/export_world.js Renderer/terrain_lab/v2/app/.local/real_map/test.biq Renderer/verification/pickup/world.csv
python3 -m unittest Renderer.native.test_profile_v2 Renderer.native.test_native_bridge_contract Renderer.native.test_scroll_damage Renderer.tools.test_renderer_dev
python3 Renderer/native/profile_v2/verify_native.py gpu
python3 Renderer/native/profile_v2/verify_native.py replay
python3 Renderer/native/profile_v2/verify_native.py matrix
python3 Renderer/native/profile_v2/verify_native.py edits
python3 Renderer/tools/renderer_dev.py full
```

The test launcher `Renderer/TEST_IN_GAME.bat` selects `pickup-r1`, enables detailed
OutputDebugStringA and bounded file diagnostics, and invokes the installed shared
checkout's standard `RUN.bat`. That path compiles/injects the matching API 14
against the installed game, so a separate INSTALL step is unnecessary for this
checkpoint. `TEST_IN_GAME.bat --check` validates routing without starting the game.
The user launches the game and supplies the existing strategic checkpoint logs.

No new Civ III patch symbol or address entry is required. Keep the sole ready
Lab step and all deferred-system contracts intact. First-map cold preparation
remains a performance limitation to measure with the user's gameplay evidence.
