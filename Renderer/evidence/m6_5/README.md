# M6.5 Connected Base-Material Evidence

## Problem corrected

The M6.3 bootstrap sampled real normalized material textures, but each Civ III tile was still an independently shaded diamond. Mixed terrain therefore produced a checkerboard of hard material borders, and retained relief next to a custom base could expose black wedges when only one logical fallback index was replayed.

## Implemented behavior

- `native/c3x_renderer.cpp` indexes the visible terrain by authoritative Civ III map coordinate and records the material disposition of each of the four valid staggered-map neighbors.
- The pixel shader derives distance to each diamond edge in local tile coordinates. A 20-percent edge band mixes half of the neighbor material at a mapped/mapped boundary, so both participating tiles converge on the same edge treatment.
- A missing or fallback neighbor contributes transparent color through the same band. Civ III's base terrain is therefore visible beneath the transition rather than meeting an opaque custom edge.
- Coordinate-seeded whole-tile brightness variation is disabled. Lighting still comes from the shared M6.4 environment, but it no longer makes every tile read as a separate rectangular patch.
- `injected_code.c` validates the DLL fallback indices and replays the complete captured Civ III staggered base sequence whenever any visible item falls back. Opaque custom interiors then cover that underlay, while retained relief/features and feathered boundaries keep the pixels they need.

## Executable evidence

`Renderer/native/native_smoke.cpp` renders two adjacent synthetic grassland/plains tiles. It requires distinct interiors, a third mixed and opaque color at the shared edge, deterministic repeated output, one reported fallback after changing the neighbor to an unmapped family, and partial alpha in the mapped/fallback edge band.

`Renderer/native/test_native_bridge_contract.py` locks the visible-coordinate neighbor table, edge-distance shader path, fallback transparency, disabled tile-level variation, and complete native-underlay replay into source-contract tests.

`m6_5_connected_terrain` builds and executes the 32-bit native smoke, runs those bridge contracts, and recompiles the injected code with `TEST_INJECTED_CODE_COMPILE.bat`.

## Deliberate boundary

M6.5 improves continuity but does not claim that the material-only flat diamonds are finished terrain. M6.6 remains responsible for all vanilla terrain families, connected relief, real water surfaces, forests/jungles/marsh, transitions, polar ice, landmarks, geometry scale, and detail fidelity. M6.7 then holds the default-game terrain visual acceptance gate before M7 begins.
