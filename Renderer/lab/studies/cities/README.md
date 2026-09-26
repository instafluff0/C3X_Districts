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
