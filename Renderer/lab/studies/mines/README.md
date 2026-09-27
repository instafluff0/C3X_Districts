# Mine Lab study

This study renders mine proposals on unchanged `test.biq` terrain. It adds mine
and era markers only to exported Lab CSV scenes, and builds an isolated renderer
copy under `Renderer/lab/out/mines/`. It does not update the source BIQ, a
production pack, a production DLL, or accepted reference images.

## Visual fidelity playbook analysis

The six normalized mine roots contain three preindustrial and three industrial
variants. Civ III eras 0–1 select the first family; eras 2–3 select the second.
The study keeps authored component transforms, mesh positions, normals, UVs,
base color and two emissive textures. All 91 non-decal source draw parts are
retained across the six variants; 275 brown ground decal draw parts are omitted.
Normal, occlusion and gloss channels remain outside this compact runtime bundle.

The source ground decals were very large brown quads. They looked detached on
relief and extended across shorelines. A matched 1.34x comparison removes
only those quads; the 1.8x version tests readability. The final 2.3x trial
keeps the no-decal choice and separates the retained mesh draw parts so each
gets its own terrain contact. It retains their authored XY layout, rotation,
normals, UVs, and materials. The six roots produce 91 retained part assets in
this trial; the extra draw calls and pack size need production measurement.
The renderer's common sun, real geometry shadows, and authored emissive
textures remain in use, with no extra shadow blob. Forest trees surround a
small inferred work clearing. The clearing, scale, and placement scores are
Lab proposals, not source metadata.

Hill and mountain sites are tested explicitly. The shared rigid mesh path
prepares mine bodies before ordinary object compilation, so terrain selection
must update its placement records there. The trial scores sampled relief and
shore clearance and penalizes river proximity. On hills, it favors a site near
tile center and the crest, and uses the visible hill surface height. On
mountains, the mine can sit at the camera-facing base. Each source part keeps
its offset from that central site but samples the terrain at its own contact
point. Authored variants remain selected by the renderer's stable tile seed.
Long rigid parts can still bridge sharply changing relief; individual contact
samples do not deform their geometry.

These choices follow `Renderer/docs/visual_fidelity_playbook.md`: retain source
form and materials first; use one scene-wide lighting basis; compose terrain,
trees, objects and shadows together; distinguish source evidence from
inference; and compare close and gameplay views on real map terrain.

## Witnesses

- `test-biq-mine-detail-study.png`: matched ground-decal control, no-decal
  control, and larger no-decal trial across coast, forest/river, mountain,
  plain, and night cases.
- `test-biq-mine-bare-comparison.png`: gameplay views against the current mine.
- `test-biq-hill-mountain-mine-study.png`: centered 1.8x control versus 2.3x
  per-part contact on inland/coastal hills and dense/wooded mountains. The BIQ
  terrain types are checked before rendering.
- `test-biq-mine-era-study.png`: the final candidate in all four Civ III eras.
  The source art supplies two distinct families, preindustrial for eras 0–1
  and industrial for eras 2–3, each with three variants.
- `grassland-closeups/{ancient,medieval,industrial,modern}-grassland-closeups.png`:
  tightly cropped, enlarged views of every complete mine variant on flat
  grassland. Individual `*-variant-*-large.png` files allow closer inspection.
  These are crops of actual off-screen Lab renders at a 256-pixel tile width,
  enlarged for inspection. The source supplies two art families across four
  Civ III eras; the Lab marker also changes the fixture's orientation between
  era rows, which should not be mistaken for a distinct authored building set.

## Earlier central-building trial

The later user-directed trial keeps only source variants 1 and 2 in each of
the two authored art families. For each, it selects the largest attached
non-decal component, which is the central building and shaft apparatus shown
in the close-ups. It preserves that component's mesh, UVs, base color and
emissive channel, recenters its ground contact, and scales it 3.0x. It removes
the accessory rocks, shelters, carts, and brown ground decals. The unchanged
Lab selector has three variant slots per family, so its third slot reuses
variant 1; no source variant 3 mesh is present in the new pack.

On hills the building anchor is exactly at tile center. Its lower source
vertices follow the sampled visible hill height, while the upper building
stays rigid above the highest sampled ground under its supports. This fitting
is an inferred Lab treatment, not an authored source deformation. On
mountains, additional front-base candidates improve visibility; shoreline
and river checks reject unsafe forward positions and retain the earlier base
site where needed. The wooded mountain remains less legible than the exposed
ridge and needs a further visibility or vegetation pass before production.

The original 3.0x witnesses are `test-biq-mine-central-grassland.png` for the two
building variants in all four Civ III eras and
`test-biq-mine-central-final.png` for inland/coastal hills and dense/wooded
mountains against the earlier complete mine. `test-biq-mine-hill-support.png`
and `test-biq-mine-mountain-front.png` retain the placement comparisons. Each
capture checks zero fallback tiles; source `test.biq` remains unchanged.

Each output frame has a neighboring `capture.json` with BIQ, scene, DLL and pack
hashes. The off-screen renderer must report zero fallback tiles. Repeat renders
are compared by exact image hash for determinism. The source BIQ hash is
recorded in `inputs.json` and must remain unchanged.

From the project root, with Python/Pillow, Node.js and the configured Windows
Lab VM available:

```sh
python3 Renderer/lab/studies/mines/study.py
python3 Renderer/lab/studies/mines/terrain.py
python3 Renderer/lab/studies/mines/closeups.py
python3 Renderer/lab/studies/mines/central.py
python3 Renderer/lab/studies/mines/mountain.py
python3 Renderer/lab/studies/mines/hill.py
python3 Renderer/lab/studies/mines/hill_support.py
python3 Renderer/lab/studies/mines/terrain_refresh.py
python3 Renderer/lab/studies/mines/hill_lift_probe.py
python3 Renderer/lab/studies/mines/hill_comparison.py
```

Generated outputs are ignored Lab artifacts. This is a visual Lab candidate;
production integration and performance validation remain separate work.

## Flatter-terrain hill size study

`terrain_refresh.py` copies the current terrain and renderer sources into a
separate Lab root, then compares 2.0x, 2.5x, 3.0x, and 3.5x main buildings on
the same inland/coastal hills and shorter mountain cases. The larger hill
buildings still intersected the foreground contour. A hill contact probe
therefore tested 0, 10, and 20 units of body lift while retaining terrain-fit
lower vertices. The 20-unit version exposed the main facade on both hills.

`hill_comparison.py` compares smaller 1.5x and 1.8x assemblies at 192-pixel
tile width. Its full columns contain every non-decal source component from
variants 1 and 2, while the matched main-only columns use only the central
component. The third runtime variant slot repeats variant 1 in both packs.
The displayed hill comparisons use the same 20-unit body lift and terrain-fit
lower vertices; `test-biq-hill-small-full-vs-main-raised.png` is the current
close-view witness. The earlier 0-lift and larger-scale sheets remain Lab
diagnostics, not approved art. The new comparison is still isolated Lab work;
it does not alter `test.biq`, the production pack, or a staged game DLL.
