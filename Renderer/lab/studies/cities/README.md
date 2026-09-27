# City layout study

This is a Lab proposal for one fixed design per Civ III culture group, era and
population tier. Each design has base, walls, capital and walls + capital
variants. Each culture's overview follows the Civ III PCX convention: four
era rows and three population columns (Town 1–6, City 7–12, Metropolis 13+).

Each population cell contains a labeled 2×2 comparison: base, walls, capital,
and walls + capital. Focused one-era sheets instead show population rows and
the four variants as columns at a larger size. All cells assume flat grassland.
Town buildings stay within one tile. This Lab candidate lets their outer wall
overhang by up to .05 tile unit so it clears the building plots. Cities and
metropolises may extend farther. The magenta ground is a PCX-style review aid, not a
terrain asset.

`layouts.json` freezes 20 culture/era recipes, each with three population
compositions. Ancient towns/cities/metropolises have 4/6/8 houses; medieval
3/5/7; industrial 2/4/6; modern 2/3/5. Later eras use taller source bodies,
while ancient settlements gain density through more smaller buildings. Houses
can move slightly as the settlement grows, but a given house or palace keeps
the same scale across population tiers within its era. Each base city has a civic
centerpiece; a capital uses the palace on that plot so the building stays
legible. Buildings and palaces share the same tile-edge facing direction.
Walls form a complete rounded ring with 16/20/24 joined segments by population
size, a gate, and four to six small towers. The walls remain below the roofs.
The source palace selectors are provenance for these proposed mappings, not
evidence that Civ III civilizations have the same architecture as Civ VI.
These are culture-group fallbacks; a future city-style profile can override an
individual civilization without adding game-specific tests to the renderer.
Some current source pools reuse the same buildings across cultures, especially
in the modern era. The recipes give those cultures distinct arrangements, but
the shared meshes alone cannot establish a distinct cultural art style. Curate
additional source models before accepting any sheet whose identity is unclear;
the Asian ancient row is a specific comparison against the supplied reference.
The ancient Asian candidate uses the full local AncientWood source intake:
golden-roof B-family houses around its matching AncientWood palace. The houses
keep rotation zero; the palace's authored platform is turned 30 degrees so its
edges follow the same grid and tile-edge angle. Larger population tiers use
more of the available ring.
The design keeps the source models' vertical proportion in the Lab pack; the
previous inherited height conversion made its roofs and palace unusually tall.
The large stone platform is part of the sourced palace and remains a visible
art choice. The source's ancient wall kit is generic rock masonry, so the
walled row remains a visual review item for this culture.

The focused one-era sheets magnify each cell for inspecting roofs, facades and
spacing. Magnification does not imply that the same details survive at game
zoom. The sheets preserve the selected source meshes and sample their original DDS
base-color channels. This software rasterizer does not reproduce the production
D3D11 normal maps, ambient occlusion, gloss, shadows, emissive response or
terrain materials. **The sheets establish composition, not Civ VI-level visual
fidelity.** A close-up and gameplay-scale render through the current D3D11
shader/material path is required before visual acceptance or sandbox promotion.
The current `test.biq` terrain replay confirms the layout is drawn by the D3D
city path. The golden roofs are more legible than the earlier dark A-family
sample, but the city still reads softer than nearby mountains at gameplay
scale. The authored house color, LEAN and gloss atlases are 1024×512; their
projected roof areas remain small. This is still an open Lab quality issue.
The staged single-building and material evidence is in
[fidelity_findings.md](fidelity_findings.md).
A map-backed headless replay is not an in-game `test.biq` acceptance image.
Do not downsample textures, decimate meshes or flatten roofs/facades to make a
layout fit. Uniform scale, placement and art selection are the available design
controls. Evaluate how much detail survives at Civ III's normal and reduced
zooms; a source model's polygon count alone does not prove a readable city.

## Curated Mediterranean medieval candidate

The focused candidate in `medieval_recipe.py` tests a more deliberate method
against the supplied walled Roman/Mediterranean reference. It is an isolated
Lab study; `layouts.json`, the runtime city pack, and reference images are not
changed. The source audition found 25 usable components in the installed
26-component Mediterranean medieval pool. One source block has no required
base-color material; the focused importer records that rejection explicitly.

The audition showed why the earlier city looked sparse: its selected pieces
were mostly isolated towers, while the same source pool also provides detailed
multi-building blocks. `flat_blocks.py` removes only the tall, below-ground
plinth geometry from those blocks for this flat-ground study. The source
architecture, paving, UVs, materials, vertex frames, and textures stay intact.
`flat_palace.py` does the same for the selected palace's two plinth geometries.
Neither derivative is a hill foundation or a replacement wall.

The recipe places one identifiable tall civic block, roof groups, and smaller
towers. Population adds fixed-scale buildings around that core. Capital layouts
have their own `capital_houses` and `capital_centerpiece`, retaining the tall
civic silhouette and placing a distinct palace in a visible court. Both use
tile-edge-aligned facades. The recipe rejects overlapping boxes, any town
building outside its tile, and any building outside the current complete wall
ring's inner clearance. The wall itself remains terrain-following and separate
from the buildings. A future accepted style can encode its own wall perimeter;
the current renderer still uses one radius per population tier.

The current Mediterranean medieval Lab candidate omits city ground decals and
elevated masonry. Its buildings use the curated source meshes, with small
trees from the existing farm assets. The compact wall sections follow terrain
per vertex, overlap at turns, and use small sourced masonry buttresses at
their joints. This keeps the visible perimeter connected while leaving the
map terrain directly under the buildings.

Use this process for other culture/era combinations: audition the entire
available local source pool; identify landmark, roof-group and infill roles;
compose and review the three population stages and four state variants;
compile an isolated pack with source normals and material channels; compare
the sheet with native-size D3D terrain replays. Treat a visually weak culture
as an art-selection or layout problem before adjusting sharpening. The focused
magenta sheet and map-backed replay are in ignored Lab output under
`Renderer/lab/out/cities/medieval-art/sheet-no-ground-tight/` and
`Renderer/lab/out/cities/test-biq/gallery/medieval-no-ground-tight-*.png`.
The replay is not an in-game acceptance screenshot.

## Two-stage ground design

Keep each culture/era/population/capital/wall recipe's building anchors, facing,
spacing, and ground footprint authored. At map draw time, form the visible
ground from the captured site's terrain material and height samples. A small
worn-earth or stone accent can strengthen lanes and courts near buildings,
then fade into the surrounding terrain. Buildings stand on individually level
plots where necessary; the ground connects and follows the intervening slope.
Walls remain a separate perimeter that follows the terrain contour. Roads,
rivers, shorelines and neighboring relief are site constraints, not inputs to
a prepainted city image.

Ground accents are deferred for this recipe. The terrain shader blends site
materials with authored height and transition masks; the city decal shader
used one fixed color texture, which read as a patch on mixed sites. The
current recipe leaves the terrain visible. If accents return, they should
sample the already-loaded site material and preserve its color, height and
specular response rather than duplicate a swatch in the city pack.

## Reproduce

From the repository root, with local normalized city, wall and palace inputs:

```sh
PYTHONPATH=. python3 Renderer/lab/studies/cities/build_layouts.py
PYTHONPATH=. python3 -m unittest Renderer.lab.studies.cities.test_designs
PYTHONPATH=. python3 Renderer/lab/studies/cities/sheet.py
PYTHONPATH=. python3 Renderer/lab/studies/cities/sheet.py --culture asian --era ancient
PYTHONPATH=. python3 Renderer/lab/studies/cities/build_wall_bundle.py
PYTHONPATH=. python3 Renderer/lab/studies/cities/build_source_frames.py
```

The sheet renderer needs Pillow. The Codex desktop bundled Python supplies it
when the default Python does not. Output is disposable under
`Renderer/lab/out/cities/design-sheets/`; `layouts.json` is the editable proposal.
The D3D map comparison compiles only an isolated candidate pack. The broad
five-culture source frames and the layout pack can be regenerated from the
locally installed source package:

```sh
PYTHONPATH=. python3 Renderer/native/city_fidelity/prepare_pack.py \
  --output Renderer/lab/out/cities/test-biq/all-designs-pack \
  --lab-layouts Renderer/lab/studies/cities/layouts.json \
  --lab-frames Renderer/lab/out/cities/source-frames/all-city-source-frames.json
PYTHONPATH=. python3 Renderer/lab/studies/cities/test_biq_gallery.py --kind flat
PYTHONPATH=. python3 Renderer/lab/studies/cities/test_biq_gallery.py --kind hill
PYTHONPATH=. python3 Renderer/lab/studies/cities/fidelity_probe.py
PYTHONPATH=. python3 Renderer/lab/studies/cities/shader_probe.py
PYTHONPATH=. python3 Renderer/lab/studies/cities/shader_probe.py --scale 2
PYTHONPATH=. python3 Renderer/lab/studies/cities/screen_size_trial.py
```

The gallery script requires an isolated `city-preview` DLL and a test.biq
terrain capture. It verifies the selected `lab-fixed-*` composition in the
renderer trace for each image. Its flat run also writes
`Renderer/lab/out/cities/test-biq/gallery/flat-contact.png` with native-size
crops for all 20 culture/era combinations. The complete AncientWood input can be
regenerated separately for the focused Asian reference comparison:

```sh
PYTHONPATH=. python3 Renderer/tools/asset_compiler/city_asset_importer.py \
  --focus-pool city/pool/asian/ancient --all-candidates --auxiliary-uvs \
  --pack Renderer/packs/CityAncientWoodCandidates \
  --report Renderer/lab/out/cities/upstream-ancientwood/source-pools-candidate.json
PYTHONPATH=. python3 Renderer/lab/shared/cities/prepare_normals.py \
  --pool asian/ancient --include-frame \
  --source-report Renderer/lab/out/cities/upstream-ancientwood/source-pools-candidate.json \
  --pack Renderer/packs/CityAncientWoodCandidates \
  --output Renderer/lab/out/cities/upstream-ancientwood/source-frames.json
PYTHONPATH=. python3 Renderer/native/city_fidelity/prepare_pack.py \
  --output Renderer/lab/out/cities/candidate-pack \
  --lab-layouts Renderer/lab/studies/cities/layouts.json \
  --lab-focus asian,ancient \
  --lab-frames Renderer/lab/out/cities/upstream-ancientwood/source-frames.json
```

No production runtime pack, DLL, native city ownership or reference image is
changed by these commands.

## Lab site adaptation

The flat layout is the canonical visual design. At a real site, capture the
authoritative tile and neighboring terrain/river/shore information. The current
Lab hill candidate samples a 9×9 grid beneath each rigid building footprint,
sets its base above the highest sampled ground, and fills the downhill gap with
a level-topped masonry retaining face. Its lower edge follows the sampled hill
shape. The masonry material and UV patch come from the normalized medieval
wall kit and repeat at a uniform 2× module size; the city source mesh, roof,
scale and facade channels remain intact. Separate wall pieces sit beyond the
building foundations and step over the hill contour without retaining masonry
underneath. These are experimental
site adaptations and require visual review on several hill shapes. A complete
legal composition still falls back to the native city if shore, river or steep
relief makes the immutable design impossible. A future local placement search
can adjust a building without changing its scale or facing direction.

Compile each candidate into an isolated Lab pack and render it with the current
D3D11 path against `test.biq` before asking for visual acceptance. Include
flat grassland, riverside, hill, mountain-edge and shoreline sites, day/night,
gameplay and close-up zooms, shadow/material checks and a deterministic replay.
Only after the user likes those map-backed examples should the recipe move to
sandbox, and production promotion requires a further accepted result. The
renderer’s existing capital and wall flags remain authoritative; labels and
UI remain Civ III-owned.
