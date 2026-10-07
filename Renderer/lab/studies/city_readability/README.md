# City readability study

Lab candidate for making the accepted culture/era/size recipes read like Civ III
cities on the map: dense, bright, one block per tile on its own ground plate.
Nothing here is promoted; the production `CityCompositionRuntime` pack is only
read.

## What the audit found

Measured on the same synthetic map, the production cities were darker than the
grass around them (mean city luminance about 85 against grass 140; 90th
percentile about 125). Civ III's own city sprites (`Art/Cities/r*.PCX`) sit at a
mean of 90–145 with highlights near 200–230 and deep shadows; they are as bright
as or brighter than the terrain. Civ III metropolises are about 1.2 tiles wide
and cover about 180% of a tile's area including their ground; the recipes were
already about that wide, but their bodies covered under half of it, with grass
between small houses and no ground plate.

The recipe runtime also accepted native city anchors without any site test, so
outskirt bodies stood on rivers, bridges and beaches.

## Candidate

`recompose.py` revises only placement data in a separate pack
(`Renderer/packs/CityCompositionLab*`, ignored local data), keeping every
selected mesh, material and facade light:

- each building grows about its own anchor until it nearly meets its
  neighbours (uniform scale; Industrial and Modern grow most);
- remaining gaps get more of the same composition's buildings;
- Industrial settlements gain smokestacks, a factory and a workshop at the
  back; Modern ones gain apartment and hotel towers, a water tower and
  (metropolis) an electronics plant. `import_accents.py` normalizes these
  installed source buildings into the local `CityAccentsLab` pack. Smoking
  accents keep off the line straight behind the palace, where their smoke read
  as the palace's own;
- capitals that shared the generic late-era palace get one palace per Civ III
  culture (`style.json` `palaces`: America, England, Spain, Ottoman, Korea;
  Babylon for Middle Eastern ancient), imported without the source plinth and
  ground planes as `flat_palace.py` does, at the old palace's centre and width.
  `palace_audition.py` draws every imported palace for choosing;
- `recolor` pulls one hue family of a model set's base texture to a target
  colour (the Industrial walls' terracotta coping to the wall stone);
- a crisp ground plate follows the buildings and a lane network, from an era
  paving texture (packed earth, cobble, grey cobble, concrete) made tileable and
  toned offline;
- ordinary bodies are marked site-optional (pack version 5);
- attached effects: flame, smoke and night-light points recovered from the
  source attachment bones (`extract_sockets.py`, palace sockets at import),
  plus an authored smoke point at each stack's mouth, capped per city by
  priority. Palace chimney smoke is off (zero strength drops a kind).

Version-5 runtime support (gated: earlier packs decode and render unchanged,
verified pixel-identical):

- `site_keeps` drops a marked body whose footprint reaches water or a
  mountain, comes within 10 source pixels of a river centreline, sits within
  the shore clearance or spans more than the composition's relief range.
  Bridges stand over channels, so they stay clear too. Forest clearing uses the
  same test, so trees close in where a body yielded;
- the ground plate fades out over water, mountains and river channels;
- the library's `look` (gain, contrast, saturation; zero is the identity) is
  applied to lit city bodies in `city_scene_material.hlsl`, with a highlight
  shoulder so whitewash and glass keep detail, damping on pale albedo
  (whitewash, white roofs) and a fade with night activation;
- effect quads: screen-aligned world quads per attached point, drawn after the
  bodies with the pack's ground-flagged effect material (read-only depth, no
  shadow). `q8_city_effect` draws a flickering flame (kept near display range
  so it does not bloom), a billowing smoke plume of nine looping puffs with a
  stem at the mouth (darker with a furnace glow at night) and a night glow,
  all stateless functions of the visual clock and a per-point seed. Renderer64
  redraws them per frame through the fresh pipeline's effect layer; that path
  compiles and is inert with the production pack in game, but the animation
  itself is unverified in game.

`style.json` holds every knob: growth per era, extents per size, infill,
accents and widths, plate margin/feather/lane/plaza/noise/period, ground
textures and `look`.

## Commands

From the repository root (Pillow/NumPy Python for building and sheets):

```sh
$C3X_RENDERER_PYTHON Renderer/lab/studies/city_readability/import_accents.py
$C3X_RENDERER_PYTHON Renderer/lab/studies/city_readability/extract_sockets.py
$C3X_RENDERER_PYTHON Renderer/lab/studies/city_readability/recompose.py
$C3X_RENDERER_PYTHON Renderer/lab/studies/city_readability/palace_audition.py
python3 Renderer/lab/studies/city_readability/study.py render before
python3 Renderer/lab/studies/city_readability/study.py render after --pack CityCompositionLab
$C3X_RENDERER_PYTHON Renderer/lab/studies/city_readability/study.py sheet before after --vertical
$C3X_RENDERER_PYTHON Renderer/lab/studies/city_readability/study.py closeups \
    --labels before after --packs - CityCompositionLab --dlls DLL DLL
```

Cases (`cases.py`, category `cities`): `city-ladder-CULTURE` (four eras by
Town / walled Town / City / Metropolis), `city-sites-ERA` (river on the city's
edges with a bridged road, river through the outskirts, coast, hills under a
mountain range, forest and jungle, road/railroad junction) and
`city-gameplay-ERA` (a dense late-game map). The Lab preview places cities from
`C3X_LAB_TILE_CITIES`; `C3X_LAB_CITY_CULTURE` picks the gameplay and site
cases' culture, and `C3X_LAB_EFFECT_TIME` pins the effect clock
(`study.animate` joins a period into a GIF). Outputs are disposable under
`Renderer/lab/out/city-study/`.

Tests: `Renderer.native.test_city_site` (site filter, plate fade, light owner
indices, legacy packs; mutation-checked) and
`Renderer.lab.studies.city_readability.test_recompose` (no new collisions,
bounds, flags, codec round trip); `test_city_site` also covers effect quads
and v5 decode of look, flags and effects.

## Known limits and follow-ups

- Units still stand over the civic centre. Keeping a small plaza clear at the
  tile centre would let a unit stand between buildings rather than on them.
- Routes on a city tile still draw through the plate; they read as streets.
  The existing per-point route fade could ease them out under the plate.
- The `cities` asset job and `test_pickup` still name intermediates under
  `Renderer/lab/out/cities/` that an earlier cleanup removed; they cannot
  rebuild the production pack. Promotion should make the frozen production
  library the recompose input instead.
