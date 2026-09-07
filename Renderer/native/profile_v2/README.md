# Production pickup integration

The user explicitly requested this port on 2026-09-06, superseding the earlier
pause awaiting production logs. **Ready for the user-run gameplay checkpoint.** The production pickup checks pass; the API 14 DLL is staged. The existing older-profile regression limitation is recorded below. Standard `INSTALL.bat`
installs the latest injected code, with pickup-r1 and detailed diagnostics selected by default. The user launches Civ III. No Lab visual approval, LQ completion, or M9–M11 work is claimed.

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
- Extensive `OutputDebugStringA` records cover capture cost,
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

## Material binding and destination color correction

The next user screenshots revealed two distinct integration defects. The pinned
archive and all 233 selected local assets verify, but the material reader searched
past JSON object boundaries. In the selected `Civ5EnvironmentSkin`, sorted
`authored_layers.desert_base` preceded the root `base_color`, so ordinary mountains
received the brown desert layer. Coast channels could also bind nested textures.
Object-scoped lookup now resolves root and explicitly selected nested channels,
without borrowing an absent channel from a sibling. Executable tests compare the
actual parser with both local packs. Matched inland/wilderness previews confirm
gray mountain bases and corrected shallows; full Lab visual parity is still open.

Earlier headless checks omitted `custom.custom_rendering.txt`, so their performance
and matrix evidence describes default packs. The pickup harness now supplies the
same selected skin/vegetation overrides as production. The corrected-skin replay
has unit redraw p95 **1.760 ms** and prepared jumps **24.6–44.0 ms**, with zero
mesh builds/uploads. Cached/cold comparison passes (zero pixels beyond tolerance,
8 total channel error); peak mesh storage is 139,103,908 bytes. Cold preparation
remains slow. These render timings exclude the final native surface copy.

A real Windows RGB555 copy of the full-color preview reproduces the screenshot's
water/sand posterization. The final blit now detects known RGB555/RGB565 DIBs and
uses world-anchored 8x8 ordered rounding. Full-color rendered/cache pixels remain
unchanged; 32-bit or unrecognized destinations keep their original copy path.
Two fixed lookup tables total 32 KiB, initialized once; existing cache budgets and
invalidation are unchanged. The pattern is stable across fixed tile jumps at both
verified zooms. Fine dither is the tradeoff for smoother gradients on a 16-bit
surface. Runtime `destination-format` OutputDebugStringA records report actual bit
depth, masks, and whether rounding is enabled; the user's exact destination format
still awaits that log rather than being inferred as proven from the screenshot.

Windows smoke tests exercise all 256 shades through actual RGB555 and RGB565
GDI destinations: maximum 8x8 mean error is 0.064/255, input stays unchanged, and
clipped copies preserve both outside pixels and the repeated pattern. Existing
32-bit byte equality remains covered. A 960x640 terrain RGB555 copy averages
2.505 ms over 20 repetitions in the VM, including diagnostics. The matched native
comparison is `Renderer/verification/pickup/color-comparison.png`; raw images,
logs and reports remain local. Use `C3X_RENDERER_PREVIEW_COLOR=1` with
`biq_preview` for the actual GDI witness. Fifty focused tests and the portable
Windows smoke pass. The tested DLL is staged for normal `INSTALL.bat`; no injected
source changed in this correction, and Civ III was not launched.

## Cold starts and minimap jumps

The next user run confirmed slow first exposure and distant minimap jumps. The
user explicitly deferred fog-edge speckling; this maintenance changes neither fog
nor the final color-conversion path. Profiling one 960x640 cold frame found 676,718
coast queries taking 6.05 seconds within 9.30 seconds of ground compilation.

Production now gathers the exact relevant coast segments once per tile. For a
query radius r and center distance d, the d+2r disk contains every possible nearest
segment. Its bounded compact search tree preserves segment/tie order, both wrap
axes, and empty-subtree/leaf certificates; queries outside the domain or overly
dense/distant coasts use the original index. No sampled approximation is added.
A lazy 9x9 authoritative tile lookup records inputs on first access. Exact point
scratch tables recycle allocations and clear generations between owners, with a
16,384-slot limit per table and uncached evaluation on saturation. Normal finite
differences evaluate only height; unrelated material/owner work is omitted. Their
measured zero cache hits justified removing the height cache entirely.

The final controlled headless comparison, with production custom skin overrides:

| Operation | Previous DLL | Optimized DLL |
| --- | ---: | ---: |
| Initial render | 11,018.9 ms | 5,358.1 ms |
| First distant minimap area | 5,813.0 ms | 2,845.1 ms |
| Second distant area | 1,831.3 ms | 1,042.2 ms |
| Cached revisits | 3.4–3.7 ms | 3.6–4.3 ms |

The initial before/after BMP is byte-identical. The minimap cached/cold comparison
also has zero channel error. Prepared fixed jumps continue to build/upload no
meshes; terrain and volcano edits match reset renders. Unit-style redraw p95 is
1.296 ms in the final minimap witness. Observed exact-point scratch is 950,272
bytes, plus the bounded per-tile coast search. Existing retained GPU/bitmap cache
budgets are unchanged, and no disk cache was added. New `query-cache` debugger
summaries report hit/miss counts and scratch bytes without per-query timers or I/O.
Run `verify_native.py minimap` to reproduce immediate distant jumps without waiting
for nearby preparation. The final build, 50 focused tests, Windows RGB555/RGB565
blit smoke, production edit and visual matrix results are recorded in
`Renderer/verification/pickup/cold-query-performance.json`.

Cold work remains noticeable; these measurements do not prove vanilla-speed
parity. The expanded **older frozen-profile** boundary test failed on both the
previous DLL (1,803 changed pixels / 50,799 error) and the candidate (1,676 / 47,872),
against unchanged thresholds. This is an existing regression, not a clean full-suite
pass; it remains open. The current pickup tests are recorded separately. No new
Civ III patch, injected source edit, or automatic game launch is involved.

## Earlier verification (default packs)

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

Run the standard `INSTALL.bat`, then launch Civ III normally. No special launcher
or environment settings are required. The latest pickup-r1 profile and detailed
OutputDebugStringA diagnostics are the defaults. Normal gameplay creates no log
file. Capture the debugger output as usual. `Renderer/TEST_IN_GAME.bat` remains
an optional convenience and also uses debugger output only.

For historical regression fixtures only, explicitly select
`C3X_RENDERER_VISUAL_PROFILE=frozen`. `C3X_RENDERER_TRACE=0` disables diagnostics.
Explicit `C3X_RENDERER_TRACE_FILE` remains available to standalone verification
harnesses only as an opt-in; production never supplies it. The headless
`verify_native.py defaults` check clears all three settings and verifies rendering
without creating or modifying a trace file. Existing custom configuration already
enables the renderer.

The user-reported execute access violation at `0xFFC179C9` was traced to a direct
`OutputDebugStringA` call in authoritative world capture. In the installed EXE,
return address `0x00DD4C82` follows a call to thunk `0x00E1B5C0`, whose slot
`0x00E32F5C` contains exactly `0xFFC179C9`. It retained an invalid installer-time
API address. The call now uses the established `(*p_OutputDebugStringA)` game
import and explicitly terminates its bounded message. A regression guard rejects
direct debug API calls anywhere in injected source. Automatic native file logging
was also removed at the user's request; it was not the identified crash cause.
Compile/replay checks do not replace the pending user-run gameplay confirmation.

The subsequent black-map report was a separate ownership failure: the DLL
successfully rendered, then the injected bridge rejected replacement flags on
caster/prefetch-only records. The previous staged DLL reproduces this at record
168 (`captured=8256`, `replacement=4`) in the headless production-boundary check.
The DLL now clears ownership only for non-RENDER records before retaining the
geometry result. Their geometry, shadows, prewarming and cache entries remain.
The strict injected validator is unchanged; tests execute its actual source body
and preview replays check ownership on every cold, warm and scroll result.

An exact inland flat-ground certificate avoids redundant height/normal samples
when the coast distance and a recorded 5×5 neighborhood prove all relief channels
zero. It includes the normal-sampling collar and records terrain/coast dependencies
so edits revoke the shortcut. The cold reference image remains byte-identical.
Initial geometry still takes seconds; this fix does not claim instant first load.

No new Civ III patch symbol or address entry is required. Keep the sole ready
Lab step and all deferred-system contracts intact. First-map cold preparation
remains a performance limitation to measure with the user's gameplay evidence.
