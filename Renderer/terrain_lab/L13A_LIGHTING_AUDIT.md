# L13A Shared Lighting Audit

Status: explicitly approved by the user on 2026-09-05.

## Candidate scope

L13A relights the approved L13 16x12 / 192-tile river scene without changing
its terrain identities, geometry, feature placement, river topology, coastline,
or alternate-skin inputs. `RUN_L13A.bat` requires
`Civ5EnvironmentSkin`, `Civ5EnvironmentVegetation`, the normalized decal and
terrain-element packs, and the normalized shore bundle. It reuses the locked
raw BIQ origin `(34,26)` and its horizontal-wrap adjacency halo.

The Lab links `native/environment_runtime.cpp` and calls the same
source-independent `evaluate_environment` evaluator used by the renderer
contract. It does not duplicate a separate Lab clock or Civ VI-specific
lighting path. The fixed evidence hours match `canonical/daynight.png` in
reading order: noon `12.0`, sunset `18.0`, midnight `0.0`, and sunrise `6.0`.
That canonical file is `1608x1368`, SHA-256
`5e210083753c9938478c636815bf50ae6d2f5d82f4fe9cd2a45c33357db65c5a`.

## Rendering behavior

The shared environment supplies sun and moon direction/color/intensity,
ambient color, exposure, shadow strength, night activation, water Fresnel and
specular scales, and the global emissive scale. Terrain normals, authored
relief, water, rivers, and renderer-owned feature bodies consume this state.
Terrain-height ray tests and height-scaled projected feature footprints supply
stable cast shadows; the accepted contact-shadow grounding remains in place.
The BIQ heightfield ray converts its inverted continuous-Y coordinate back to
the shared feature/local-v light basis, so hills, mountains, volcanoes, trees,
and other raised bodies all cast along one screen-space direction per frame.
Projected feature vertices follow the terrain receiver's screen-depth gradient,
so shadows are not hidden as they extend down-screen. Forest, jungle,
mountains, volcanoes, hills, and renderer-owned river/shore bodies all
participate according to their actual raised geometry; flat land receives
shadows but does not invent raised occluders. Every raised class supplies both
normal-driven self/face shading and a neighboring cast shadow. Terrain relief
uses a filtered height-field ray while discrete vegetation/shore bodies use a
soft tapered projection; neither cue substitutes for the other.
Mountain, volcano, hill, forest, and jungle face contrast uses the same raised
form-response curve. Vegetation retains its authored mesh normals, with a
map-scale lighting adjustment that expands the horizontal canopy-normal
component so dense stands still show a readable lit side and opposing shaded
side at noon.

Low-angle terrain occlusion uses a soft bounded visibility term, and the Lab
applies restrained tone mapping only in L13A modes. This prevents the first
candidate's broad dark terrain bands and clipped noon sand while preserving
the approved source materials. Cast-shadow projection is deliberately
stylized: light direction rotates with the clock, while a fixed projection
slope makes height—not sun elevation—the source of footprint length. This
keeps the four canonical phases in one tight length band. The 18:00
ambient/direct response is warmer, 06:00 remains cooler, and midnight is
blue-weighted but readable. Water remains bounded and spatially varied at
night rather than becoming a uniformly bright blue plane.

No city, wonder, district, unit, light fixture, smoke, fire, bloom, weather, or
animation is introduced. A non-visual generic emissive-channel diagnostic
proves policy activation through the shared evaluator: noon reports emissive
`0.0000`, sunset and sunrise report `0.4000`, and midnight reports `1.3500`.
Each Lab process renders one immutable frame and reports
`static_redraw=idle`; runtime invalidation remains a Game Integration concern.
Visible night lights are deliberately deferred to L17, where real city window,
lamp, emissive, and analytic-light attachments can prove the presentation.
Later improvement gates may add only lights owned by their source-backed
improvement assets. L14 roads inherit the approved night environment and do not
invent streetlights.

## Promotion matrix and determinism

`RUN_L13A.bat` emits the same 192-tile composition at native 3200x1800 and a
true box-filtered 1600x900 Civ III reduced zoom for every fixed hour:

- `terrain_beauty_l13a_noon.bmp`
- `terrain_beauty_l13a_noon_zoom2.bmp`
- `terrain_beauty_l13a_sunset.bmp`
- `terrain_beauty_l13a_sunset_zoom2.bmp`
- `terrain_beauty_l13a_midnight.bmp`
- `terrain_beauty_l13a_midnight_zoom2.bmp`
- `terrain_beauty_l13a_sunrise.bmp`
- `terrain_beauty_l13a_sunrise_zoom2.bmp`

Two consecutive executions produced byte-identical outputs. SHA-256 values:

- noon: `379ac9ff2effc36654a2ad1118b6c67f1de2fc140cfeff013172aac37617a920`
- noon reduced: `d1c4cc3974833722827bd717d0134a5d858b6721c724812b53f47a6713d575bb`
- sunset: `08b1a6582346e2df9ededbb8f9732482743492e158b109e5221bb915eb4ed95b`
- sunset reduced: `6cebf187edb02e03b0a6562196fb9f2b27e4d1e3cde7c265c04c9be00e0f7474`
- midnight: `e5f7a62491d877430cd83d2ee9fecb75556243d1cf52bc02c6b330ff2bc4d8dc`
- midnight reduced: `0e2993b5faedc05bdc3dcf89deb41e6e355e72c723a4430f557ab8f2280b51cc`
- sunrise: `d2042d5d436db1f327a6c8b4ac3476d63528b4b8afe9b4d26cfeb3cf88493c29`
- sunrise reduced: `ee592df81e611319ea1923fb27a05f0b940622104f7286fa11917dc92e04e326`

The source viewport CSV remains byte-identical to approved L13:
`1500c085e88425796fb150db4a96a904722d1f8b0cce1d614d46db58bc3c0b4b`.
The official `renderer_dev.py lab` path passed 62 Python tests, 12 BIQ/exporter
tests, the Windows build, and all eight renders on 2026-09-05. The user approved
L13A on 2026-09-05; `handoffs/L13A_lighting.json` is the frozen source of truth.
