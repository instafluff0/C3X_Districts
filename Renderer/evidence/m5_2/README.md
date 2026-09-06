# M5.2 Live Scene Evidence

Status: accepted under the lightweight development validation policy requested by the user.

## Implemented Capture Path

- With custom rendering enabled, the first successfully composited frame exports automatically; `Ctrl+Shift+F12` optionally requests a recapture.
- `Renderer/native/scene_export.cpp` writes `Renderer/validation/live/civ3-live.scene.json` atomically.
- The scene contains authoritative tile anchors, viewport/environment state, terrain records, and captured instances for every renderer category present in the selected view.
- Terrain remains the only replacement-owned category. Other categories are capture-only facts and continue through Civ III's retained draw paths.
- The portable `m5_2_scene_export` gate proves strict schema validation, deterministic output, all-category synthetic coverage, offline batch behavior, and injected-code compilation.

## Automated Evidence

- The strict native smoke fixture round-trips through `c3x.visible_scene.v0` and contains all nine renderer categories.
- Repeated native exports are byte-identical.
- Offline batch tests cover canonical scenes, category inventory, deterministic PNG/report output, hashes, mapping and bounds metrics, and honest pending/review state.
- `TEST_INJECTED_CODE_COMPILE.bat` passes.

## In-Game Health Evidence

`game_health_full_ui.png` was supplied by the user after exercising the custom-rendering build. It shows the diagnostic terrain beneath Civ III-owned features, city, unit, selection ring, borders, fog/shroud, labels, minimap, HUD, and full-screen UI at 4450x2364. The game compiled, ran, and no crash or odd behavior was reported.

SHA-256: `986196524F3D27D41C9D3D3BA3CEBE1C06A7E0E05F75FC47CA9B301423F86EA6`

Per the user's explicit direction, routine renderer development keeps manual validation simple: screenshots and reported behavior are sufficient as the game-health signal, while reproducible structural guarantees remain automated. Formal paired fixture curation can return for release candidates or when a visible regression needs diagnosis.
