# Wonder survey (Lab)

Preparation for renderer M10: which Civ VI or Workshop wonder art each default
C3X wonder (the 23 in `default.districts_wonders_config.txt`) would use on the
Renderer64 map, and how it looks while it is being built. Nothing here is
promoted; production packs, ledgers and the renderer are unchanged.

- `kits.py` imports each source into its own Lab sub-pack
  (`Renderer/packs/WonderKitSourcesLab/<key>/`, so imports run in parallel),
  bakes catalogs into `Renderer/packs/WonderKitCatalogLab` and renders them
  through the district study's Lab path (`Renderer/lab/studies/districts/kits.py`).
- `stages.py` poses a wonder's Civ VI build movie at any build progress.
- `plan.py` holds the picks (recommended first), stage rows and city scenes;
  `content.py` the page prose and Workshop candidates; `page.py` builds
  `Renderer/lab/out/wonders/page/`.

```sh
python3 Renderer/lab/studies/wonders/kits.py compile --jobs 3 [--source KEY ...]
python3 Renderer/lab/studies/wonders/stages.py prepare --source KEY ...    # VM: convert the build movie
python3 Renderer/lab/studies/wonders/stages.py probe --source KEY --built --at .25 .5 .75 1
python3 Renderer/lab/studies/wonders/kits.py render [--catalog NAME ...]
python3 Renderer/lab/studies/wonders/page.py
```

## Findings worth keeping

- Civ VI keeps constructed wonders only in `WonderMovie.blp` (base, each DLC
  and Workshop mods alike). The model is the build movie: every block, brick,
  scaffold pole and crane is a rigid mesh on its own bone; props (cranes,
  brick and wood piles, palms) attach to bones.
- The importer keeps each draw's mesh index. The mesh -> bone table is the
  base model's `int32` pointer field whose element count equals the mesh count
  (FGX behaviour data next to the `MeshBinding` records).
- The movie (the base model's `AnimationDesc` entry in the package's
  `BLP::AnimationEntry` table) converts with
  `Renderer/tools/asset_compiler/CONVERT_CIV6_ANIMATION.bat` at 0.01. Most bone
  tracks are constant: pieces appear and disappear through per-bone
  `<bone>_visibility` Granny vector tracks, which the Firaxis managed wrapper
  does not expose. `export_visibility.cs` reads them from the wrapper's native
  `granny_track_group` (packed: VectorTrackCount at 8, VectorTracks at 12; a
  vector track is 32 bytes: name, key, dimension, curve) and evaluates them
  with Granny's own `GrannyEvaluateCurveAtT`, reached by reflection on the
  C++/CLI `<Module>`.
- A piece's pose is its bone's inverse bind times the sampled world matrix;
  hidden or zero-scaled bones drop the piece. The movie's last sample can
  evaluate nearly empty; the frame before it is the finished wonder, free of
  scaffolds (no per-wonder cutting).
- Movies build two ways: some raise the finished pieces (Pyramids), others
  build with construction pieces and swap the finished building in at the end
  (Great Library). C3X build progress maps onto the span from 10% of the
  busiest frame's piece count to the frame from which the finished set shows.
- Lab importer tolerances (`kits.compiler`, this process only): the package
  model's string array whose count equals the bone count; a placeholder for an
  unnamed bone (the Oracle's); the district study's build-movie bind-pose
  relaxation looked up at call time.
- Composition fit: the main model's raised pieces (flat ground pieces and
  props excluded) span `fit` (1.0) tile, capped at `height` (1.2); construction
  stages share the finished wonder's fit and centre.
- District buildings used as stand-ins have no build movie; their stages rise
  from the ground (cut at the built share of their height).

## Night lights and Civ VI effects

- Wonders are drawn as Renderer64 city bodies (`city_pack.py`), not resource
  bodies: Civ VI maps a building's emissive (night lights) with its third UV set
  and AO with its second, as production cities read them, and resource bodies
  carry one UV set. Sources import with `auxiliary_uvs` (kits.compiler);
  `kept(full=True)` keeps one group per material with all three UV sets.
- Pitfalls found on the way: merging pieces per texture also merged their
  materials (one lit material's emissive spread over every piece sharing its
  colour: orange stonework), and sampling emissive with uv0 reads the colour
  layout. Many build movies swap the finished, lit building in on their very
  last frame (Temple of Artemis); `stages.last_frame` keeps that frame unless
  it shows under half the vertices of the one before.
- The Lab city library (`Renderer/packs/WonderCityLab`) is production's
  (the runtime needs its complete city set) plus one composition per cell,
  authority `wonder:<cell name>`; production textures are hard-linked into it.
  The Lab resource pack keeps each cell's decals and a `clearance/` marker that
  names the cell's tile.
- Fires and lights: a WonderMovie model's `TimelineSet::Trigger` records (28 B:
  kind 1 = spawn, start time, attachment index, FNV-1a of the attachment bone's
  name at 0x10, FNV-1a of the effect name at 0x14) start VFX/Light package
  entries. `effects.py` keeps the idle loop and permanent triggers, resolves the
  bone by its name hash and classifies by name; `kits.effect_records` lays them
  out with the bodies as city effect records: flames (0), smoke (1), lamps (7)
  and blinkers (8). 90 + kind stays below 100, where body materials start.
- `effects_tree.py` builds the private native tree: wonder compositions are
  selected on their cell's tile (never by cities), resource tiles are
  prepared, lamp kinds accepted and the library cap raised to 512 MB.
- Not covered: brazier bones of Civ VI's separate finished-wonder model that
  the import lacks (Great Library, Hanging Gardens, Statue of Zeus braziers);
  waterfall mist.
