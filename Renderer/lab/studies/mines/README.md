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
```

Generated outputs are ignored Lab artifacts. This is a visual Lab candidate;
production integration and performance validation remain separate work.
