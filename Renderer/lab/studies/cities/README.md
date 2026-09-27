# City layout study

The five Civ III culture groups are American, European, Roman, Middle Eastern,
and Asian. The renderer's internal `mediterranean` slot is the Roman group;
it is not a sixth culture. The local Civ VI import strategy maps its art
families into these five slots offline.

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

`layouts.json` holds the original 20 culture/era baseline recipes. Its ancient
4/6/8, medieval 3/5/7, industrial 2/4/6 and modern 2/3/5 house counts are
not a size target for the focused candidates. Use the medieval Roman city's
roof and body size as the visual target for ancient, medieval and industrial
art, even when their historical sizes would differ. Increase population by
adding buildings rather than enlarging existing ones. Houses can move slightly
as the settlement grows, but a given house or palace keeps the same scale
across population tiers within its era. Each base city has a civic
centerpiece; a capital uses the palace on that plot so the building stays
legible. Buildings and palaces share the same tile-edge facing direction.
Walls form a complete rounded ring with 16/20/24 joined segments by population
size, a gate, and small buttresses over every joint. The walls remain below the roofs.
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

The other medieval cultures now have focused Lab candidates in
`medieval_block_recipe.py`. The full source auditions found 23 usable European,
31 Asian, 38 American, and 37 Middle Eastern pieces. Earlier generic sheets
mostly chose isolated, narrow buildings even though each pool has complete
neighborhood blocks. `flat_blocks.py` identifies deep plinth meshes in
each block by vertical span rather than source order, then
omits only those draw bindings in a local derivative pack. The selected palace's
separate below-ground plinth bindings are also omitted. Source roofs, facades,
materials, normals, and horizontal scale are retained. Each culture's offline
profile selects a large central block, substantial flank groups, readable
individual landmarks, and a matching palace. The curated town starts with five
surrounding pieces, like the Mediterranean candidate; city and metropolis add
larger roof groups and infill to suit each source pool. Growth adds buildings
without resizing existing ones. The magenta sheets are
under `Renderer/lab/out/cities/medieval-{european,asian,american,middle_eastern}/`
and do not change `layouts.json` or any production pack.

The current medieval review keeps all five Civ III culture-group candidates
separate. Roman City/Metropolis layouts have 15/21 surrounding buildings;
European, Asian, and Middle Eastern have 16/22, while American uses 13/19
larger neighborhood pieces. Capital layouts have their own checked infill, and
farm-source trees occupy clear plots. No ground decal or elevated masonry is
included. European's source `LG_SQ_01` and `SQ_02` blocks contain conspicuously
misaligned houses baked into single meshes. This candidate selects the more
coherent `LG_SQ_02` core, removes `SQ_02`, and substitutes larger, consistently
facing houses for its two small side pieces. The full 12-cell sheets are under
each culture's `review-dense-sheet/`; native map replays are under
`Renderer/lab/out/cities/test-biq/gallery/medieval-*-review-dense-*.png`.
These remain Lab examples for visual review, not promoted art.

The optional, ignored `medieval-source-families/` intake preserves all 23
installed `ARTERA_CLASSICAL` source culture tags and 676 usable components,
including families beyond the five provisional Civ III mappings. That source
catalog is an offline asset-pipeline input; game drawing still selects generic
pack metadata without source- or civilization-specific branches.
`ARTERA_CLASSICAL` is Civ VI's shared visual tier for its Classical, Medieval,
and Renaissance gameplay eras, as recorded in the installed `Eras.artdef`.
"Medieval" here names our Civ III Middle Ages **target row**, not the historical
period of every source building. See [the source-era inventory](source_era_inventory.md)
for the era mapping and tag counts.

For a complete medieval-family audition, `medieval_family_review.py` creates
one isolated 12-cell layout per installed source tag. The five source tags
already used by the focused Civ III candidates reuse their curated layouts;
the other 18 use the same measured-size, collision-checked initial placement
method and their matching source palaces. `medieval_family_palaces.py` removes
separate palace plinth and ground-plane bindings where present.
`medieval_family_build.py` generates each full magenta sheet and a source-frame
backed candidate pack, and `medieval_family_maps.py` runs five headless
`test.biq` examples per family. The index and comparison atlases are under
`Renderer/lab/out/cities/medieval-source-families/review/`. The first-pass
layouts are for choosing which families merit hand calibration; a source tag
is not a hard-coded runtime civilization or a new Civ III culture slot.
The sheet renderer restores each wall part's source center after loading its
centered preview mesh, so the preview ring matches the connected native ring.
The larger metropolis cell is framed higher to keep its near wall visible.

The all-era source-art auditions now use separate foundation-free Lab packs.
`trim_subsurface.py` clips imported city-component triangles at the authored
ground datum, preserving UVs, normals, materials and the original source pack.
`foundation_grades.json` records the few visually calibrated cuts where a
visible masonry pedestal rises above that datum. The American comparison
replaces mixed-facing source blocks with aligned individual houses; Baltic,
Scottish and Vietnamese civic centers likewise use aligned source singles.
Palace roots use their measured footprint correction so their fronts match
the SE/SW street grid. These edits affect review assets and recipes only.

## Ancient culture auditions

The focused ancient import now retains all seven ancient source tags in a
source-neutral local pack: AncientEarth, AncientWood, Babylon, Cree, Gaul,
Mapuche and DEFAULT. Gaul's 33 resolved city entries duplicate AncientEarth's,
leaving six distinct component sets to compare. The older five-culture fallback
reused AncientWood for American and Asian, AncientEarth for European and Middle
Eastern, and DEFAULT for Roman. The current auditions instead try Mapuche for
American and Babylon for Middle Eastern, with Cree as an additional alternate.
These choices are offline mappings for review, not runtime civilization tests.

`ancient_roman_scale_recipe.py` arranges complete source blocks at fixed scales
around a central civic or palace; `ancient_babylon_recipe.py` arranges the
Babylon singles because that source family has no complete blocks. Each recipe
checks town bounds, body overlap and clearance inside the wall ring. The
current auditions add buildings across population tiers rather than enlarging
existing roofs: the block families use 5/17/22 houses and Babylon uses
6/18/26. The city and metropolis tiers have roughly 30 percent more buildings
than the preceding review, placed within the same wall ring. Capital and
walls + capital now use additional, independently checked palace-side infill
plots, so replacing the civic centerpiece does not leave the capital variants
visibly thinner. The two capital states share the same building list; only the
wall state differs. AncientWood and
Babylon use their own vertical conversion to keep
their taller source silhouettes in proportion with the medieval Roman review
target; horizontal scale and source roof detail are retained. The
palace shares the tile-edge facing with the houses. Local derivative packs
omit the deep source plinths and the separate, low horizontal ground planes in
both neighborhood blocks and palaces. The palace's modeled architectural
footing remains part of its building mesh. No city ground decal, runtime paving,
or elevated masonry is included. A Lab-only wall bundle substitutes a complete stone span
for the ancient kit's open two-post gate so the native replay has a closed
perimeter.

The older seven-family source sheet at
`Renderer/lab/out/cities/ancient-source-families/family-sheet-flat.png`
predates ground-plane removal. The current four-state/three-size sheets are under
`Renderer/lab/out/cities/ancient-{american,european,mediterranean,asian,middle_eastern}/roman-scale-sheet-v11/`;
Cree uses `ancient-source-families/cree-sheet-v11/`.
The seven tagged source families are shown together, including Gaul's
AncientEarth duplicate, at
`Renderer/lab/out/cities/ancient-comparison/ancient-all-families-magenta-v11.png`.
The matching town/city/metropolis native-size map comparison with the medieval
Roman reference is at
`Renderer/lab/out/cities/ancient-comparison/ancient-all-families-test-biq-v11.png`.
These map shots use captured `test.biq` terrain and the D3D renderer, with
trace-verified Lab compositions and no renderer fallback. They are not live
Civ III screenshots or approval images. Retain all families as candidates
until their appearance is reviewed.

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
Lab candidate places each rigid building above its sampled footprint and leaves
the terrain directly beneath it visible. The earlier elevated retaining masonry
and city ground decals are absent from these candidates. Separate wall pieces
follow sampled terrain per vertex around the buildings. This adaptation still
needs review on several hill shapes, especially where a steep local slope could
leave a visible gap beneath a rigid building. A complete legal composition
falls back to the native city if shore, river or steep relief makes the immutable
design impossible. A future local placement search can adjust a building
without changing its scale or facing direction.

Compile each candidate into an isolated Lab pack and render it with the current
D3D11 path against `test.biq` before asking for visual acceptance. Include
flat grassland, riverside, hill, mountain-edge and shoreline sites, day/night,
gameplay and close-up zooms, shadow/material checks and a deterministic replay.
Only after the user likes those map-backed examples should the recipe move to
sandbox, and production promotion requires a further accepted result. The
renderer’s existing capital and wall flags remain authoritative; labels and
UI remain Civ III-owned.
