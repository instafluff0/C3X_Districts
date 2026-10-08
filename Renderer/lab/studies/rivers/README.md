# Rivers

Accepted by the user and promoted to the game on 2026-10-07. The game reads
the river shading from the pinned `Renderer64ResidentRuntime` pack; after any
river shader change, rerun `Renderer/tools/overlay_river_shading.py`
(`--dry-run` first, a new dated `--backup` each time).

## Changes

- **Banks never cover objects.** The river layer (ground kind 9) wrote a depth
  pulled about 27 units toward the camera across its whole bank band. Its
  ground basis weighted height far less than the natural terrain (0.35 against
  1.73 depth units per unit of relief), so it needed that bias. Bridges, farms,
  routes and object shadows drawn after it failed the depth test there; trees
  drawn before it were painted over. Now the generated hydrology shaders define
  `C3X_RIVER_NATURAL_DEPTH`: river vertices sort on the natural basis with a
  0.004 x H bias. The river draws with the routes' test-only decal depth state
  in the Lab (`submit_geometry`) and the game (`fresh_pipeline.h` `draw_layer`;
  the overlay layer's water prepass no longer draws it).
  Tests: `Renderer/native/test_river_depth.py`.
- **Ocean optics.** The river material uses the in-game sea's response:
  Fresnel-weighted sky, the mirrored scene (lifted by the river's height above
  the sea plane, faded on raised reaches) and the open-sea sun and moon glare.
  The glare uses a calm normal so it reads as a smooth sheen.
- **Objects and mountains only.** Rivers reflect standing objects (forests,
  jungles, units, buildings) and mountains, never other ground: land just above
  the sea plane mirrored onto the rivers beside it. Mirror passes draw ground
  (relief, terrain decals, farm fields) with the color-only
  `reflection_terrain_blend`, so the mirror's alpha counts objects; mountains
  keep alpha (the game relights them with the ground, then redraws them with
  the alpha-only `reflection_coverage_blend`). Hills stay ground. The seas
  count mirrored ground color as coverage and keep their coastal reflections.
  A mirror image keeps a quarter sky haze and outweighs the water's dark body
  (weight .9 vs .272 for open sky), so a shaded mountain face reads as soft
  grey instead of near-black.
  A mountain's grassy foot mirrors like ground, so its mirror coverage eases
  in with height (smoothstep 10→30 units in the generated mountain
  `PSReflection`, from `environment_refresh/prepare.py`); peaks still reflect.
  Rivers mirror mountains and volcanoes at half strength: the coverage blend
  takes its alpha from `mountain_mirror_coverage` (.5) in both the Lab and the
  game, since a tall peak beside a narrow river broke into pale patches.
  Tests: `Renderer/native/test_river_reflection_objects.py`.
- **Stream lines.** Thin pale lines parallel to the banks, broken into dashes
  that drift downstream along the drainage tangent; a broad world field picks
  occasional livelier reaches. Headwater pools stay calm.
- **Natural banks.** The waterline wanders between the ground grid's samples.
  A translucent dark layer darkens the land beside the water instead of
  painting a soil ribbon, and authored river gravel collects along it.
  Shallows keep the channel color, so no pale rim outlines the water.
- **Delta.** At a mouth, two narrower distributaries leave the last reach and
  fan out to the sea beside the main outlet (`river_corridor.h`).
  Tests: `Renderer/native/test_river_delta.py`.
- **Bridge waterline.** Bridge models continue below their base (a quarter of
  the stone arch, half of the industrial and modern piers). The old river
  depth hid that part; without it the bridges looked tall and off-centre. A
  bridge's base (its lower bank) is the waterline: VSSharedFeature carries the
  height above it in q6_world.w (2 + h/128) and PSIntegratedFeature clips
  bridge fragments below it. Tests: `Renderer/native/test_bridge_waterline.py`.

## Lab

```sh
python3 Renderer/lab/studies/rivers/study.py render after
python3 Renderer/lab/studies/rivers/study.py render after --hour 18
python3 Renderer/lab/studies/rivers/study.py motion after
$C3X_RENDERER_PYTHON Renderer/lab/studies/rivers/study.py sheet before after
$C3X_RENDERER_PYTHON Renderer/lab/studies/rivers/study.py gif after
```

`--dll` pins a candidate copy. `--shader-root NAME` renders with a complete
private shader tree under `Renderer/lab/out/river-study/NAME`, so tuning never
changes the checkout's generated shaders that other sessions render with.
The case lives in `cases.py` (category "infrastructure"): a meandering river
through farmland with bridged road and railroad, hills and a mine on the far
bank, forest on both banks, resources and a coast.

The rivers category's own water-motion lifecycle witness currently fails on
the unchanged checkout too (only the first of 48 frames changes); the study
keeps those stills and records the failure.

## Game

`overlay_river_shading.py` copies whole regions into the pack (the
`C3X_RIVER_NATURAL_DEPTH` define, `translated_depth`, the
`Q3_CONTINUOUS_RIVERS` branch of `q3_water_material`, and the two bridge
waterline lines); `river-overlay.json`
records each overlay and its backup. The delta and draw state are ordinary
C++, so a Renderer64 build picks them up. The 1498 AD `near` scripted test
(`Renderer/.cache/river-integration/`) shows whole bridges, farms stopping at
the banks, stream lines and the softer banks.
