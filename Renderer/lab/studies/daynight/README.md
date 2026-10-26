# Day/night Civ VI match (Lab study)

Goal: C3X's clock lighting should read like the user's four-panel Civ VI
reference (noon, 18:00, midnight, 06:00, upper left to lower right): never
too dark, bluish at night, a beautiful gold sunset. The reference is kept
locally as `Renderer/lab/out/daynight/civ6-reference.webp` (not committed).

## Findings (2026-10-24)

- Civ VI's `GameLighting.artdef` is plain XML with hour-keyed sun colour,
  exposure, light-rig and colour-grade weights (see
  `docs/civ6_lighting_findings.md`). Sunrise is a dim lavender key light,
  sunset a bright gold one, night a weak blue key lifted 1.5 EV.
- The old curve dimmed sunset to a quarter of noon's sun and lowered exposure,
  so 18:00 was dull olive (lightness 0.66 of noon against Civ VI's 0.94), and
  midnight was murky olive (0.52 against 0.74).
- Relief, mountains, trees and units (BeautyFrame receivers) skipped the
  gameplay night lift (ambient x1.8, moonlight x2.2*.82) that cities, water and
  rivers get, so at night most of the land had about half their light.
  `test_source_night_response.py` now holds them together.
- The game's output transfer is half Reinhard, half filmic fit
  (`resident_scene.cpp`); Lab previews use plain Reinhard. Judge brightness
  through the study's private DLL, which uses the game transfer.

## How it works

`study.py build --overlay DIR` snapshots the working tree into ignored
`Renderer/native/build/daynight-src`, overlays the proposed production files,
makes the curve's key tables loadable from `C3X_LAB_DAY_NIGHT_KEYS`, switches
the Lab output to the game transfer and builds a matching `native_preview.exe`
(the tree's frame ABI can be newer than the shared preview's). `build` alone
makes `daynight-baseline`, the working tree with only the game transfer.
`render LABEL [--keys FILE] [--season N] [--shader-root DIR] [--views ...]`
renders `concept-HH` or `village-HH` views (season 0 Summer, 1 Autumn, 2 Winter,
3 Spring). Seasonal ground, water and farms exist only in shader roots made by
`python3 Renderer/tools/season_shading.py --lab-root DIR`; without one, a
non-Summer render shows only the seasonal tree stands;
`measure LABEL...` prints OKLab lightness, hue and chroma of grass, tundra and
the whole view against the Civ VI targets; `timelapse LABEL` crossfades the 24
hourly `concept` renders into a looping MP4 with a clock bar.

`build --head` snapshots tracked sources from HEAD instead of the shared working
tree (other sessions' uncommitted edits stay out). Every private build also
carries a Lab-only flurry patch: Renderer64 draws the Winter mountain flurry as
per-view quads in its x64 pass, which Lab previews never run, so with
`render --flurry` (`C3X_LAB_MOUNTAIN_FLURRY`) both Lab object paths attach the
same quad as a city effect, and the season root's shader draws it in Winter.
`render --minute M --time T` places a frame at M minutes past each hour with
animation clock T, so films step in quarter hours and their water moves.

The x64 sandbox client (`Renderer/sandbox`) cannot stand in for the game here:
it never establishes the game's world window, so rivers, roads, farms, sites
and mine bodies get no instances (hut shadows without huts). Films of the Lab
village use the private Lab build.

Looks: v1 first plausible keys; v2 keyed exposure and saturation; v3 more
orange sunset, brighter bluer night (the user's pick for midnight: it matches
Civ VI). v4–v7 tried calming the pale-blue glow of sand, snow and shores at
night (a blue tint grade with neutral light, then silvered highlights); each
lost the vibrancy and was dropped. v5 introduced the rose-gold golden-hour
sunrise the user asked for (Civ VI's dim lavender made units hard to see),
which stays.

Lab coast blotches (2026-10-25): dark patches offshore in the village view. The
game's mirror-ground clip (84c7115f, one height unit) now also holds in the
Lab's terrain and mountain mirrors (`environment_refresh/prepare.py`), but that
only lightened them. A mirror tinted per provider (each `PSReflection` painting
any covered pixel one solid colour) showed the rest was the terrain mirror's
faint coastal margin: the land fades to a few percent over the beach, the
ground blend writes colour without alpha, and the sea (`water_natural.hlsl`)
takes any mirrored colour as full coverage, so over open sea that margin read
as a near-black solid. The Lab terrain mirror now keeps only ground the main
pass draws at least half opaque, at its own colour. They showed most away from
the screen centre, where the sea's Fresnel is strongest.
