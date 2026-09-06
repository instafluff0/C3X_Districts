# L14 Road Audit

Status: complete; deterministic 192-tile alternate-skin candidate explicitly approved by the user on 2026-09-05.

## Scene and provenance

- Terrain input remains the unchanged authoritative 16x12 BIQ viewport exported from installed `Intro1 Ancient Treasures.biq` at raw origin `(34,26)`.
- `fixtures/l14_roads_192.csv` is a separate deterministic, source-independent Lab augmentation. It is labeled `lab_augmentation`, records the BIQ input SHA-256, and is never represented as captured Civ III road state.
- The candidate contains 98 connected road nodes and 109 edges, including 32 junctions, 14 intentional ends, multiple cycles, 17 exact shared-edge river crossings, relief traversal, four route stages, pillaged coverage, and six horizontal-wrap continuations.
- Ground and vegetation use only `Civ5EnvironmentSkin` and `Civ5EnvironmentVegetation`. Roads sample the normalized route material atlases directly. Bridges use normalized medieval, industrial, and modern rigid bodies and source textures through the generated source-independent `bridge_runtime.bin` bundle.

## Rendering contract

- Road topology is one continuous centerline graph, not isolated per-tile sprites.
- Each edge is subdivided and draped over the accepted continuous terrain heightfield. A restrained deterministic lateral curve breaks up ruler-straight center-to-center runs, while its offset returns exactly to zero at both graph nodes so every shared endpoint and bridge mouth remains aligned.
- Source tiled-path endpoints extend through graph nodes with bounded overlap. A narrow source-colored centerline coverage guard closes transparent atlas pinholes without procedurally repainting or widening the authored road material.
- Road strips use the source-authored atlas coverage and route stage layers. No procedural road color, smoke, traffic, animation, light, rail, sleeper, or streetlight is added.
- Bridge placement tests the road edge against the exact reciprocal BIQ river bits. A river merely existing in either adjacent tile is insufficient.
- Roads and bridge bodies inherit the approved L13A noon environment. Visible night fixtures remain deferred to L17 cities.

## Candidate outputs

- `terrain_beauty_l14_roads.bmp`: complete medieval-stage road network over the accepted scene.
- `terrain_beauty_l14_roads_zoom2.bmp`: reduced Civ III zoom readability view.
- `terrain_beauty_l14_roads_no_roads.bmp`: regression control.
- `terrain_beauty_l14_roads_only.bmp`: topology and bridge isolation.
- `terrain_beauty_l14_roads_styles.bmp`: ancient, medieval, industrial, modern, normal, and pillaged source-art isolation.

Two complete `python3 Renderer/tools/renderer_dev.py lab` runs produced byte-identical BMPs:

- complete: `038328159315349f56e86090bc491ba964d8ba84a8508b34bcd78c12d3bbf93f`
- reduced: `d57f67e67a152354dd0862b86e68c080f849cedaba2eb784ad0aa753a957eeb3`
- no roads: `379ac9ff2effc36654a2ad1118b6c67f1de2fc140cfeff013172aac37617a920`
- roads only: `9d7a63e956255d454bff454221f22d4c5553487f52e4f86e68e6fcd05a61d02c`
- style isolation: `a13f2f40e386079830c646b4c1a9003534be9851484475c51aaf11863aacdf08`
- Lab scenario: `e9bef85d65543e30bf7c1caade36cdd5b69ed097f984d8ddb75148bee70e88e2`
- bridge runtime: `8908785abd8eb87e1a790c10a05fab6e75524e0852f7270dc0ed69d38ef01797`

The no-roads control is byte-identical to the approved L13A noon render. The user explicitly approved this promotion after the connected, gently curved revision.
