# District kit survey (Lab)

Preparation for renderer M11: which Civ VI district, building and improvement
art each default C3X district would use on the Renderer64 map. Nothing here is
promoted; production packs, ledgers and the renderer are unchanged.

- `kits.py` reads every installed `Landmarks*.artdef`, picks the base variant
  Civ VI would choose for a building set and art era, compiles it and its hero
  buildings into `Renderer/packs/DistrictKitSourcesLab`, bakes Lab resource
  compositions into `Renderer/packs/DistrictKitCatalogLab`, and renders them
  through the night tree's DLL once built (else a pinned copy of the candidate
  DLL). Context scenes place production cities beside the districts through
  the Lab tile-overlay path.
- `night_tree.py` keeps a private native tree (`Renderer/lab/out/districts/tree-night`,
  mountains study `private_tree.py`) that lights resource-pack bodies at night:
  bake writes `textures/emissive_N.dds` with `atlas_N.dds`'s cell layout (Civ VI
  emissive maps, BC1, resampled to each colour cell; unlit cells black), the
  DLL flags bodies in a lit slot with material code .002 (never `decal/`) and
  binds the emissive atlases in the road slots t98-t105 for the feature draws;
  `terrain_scene.hlsl` (hand-maintained, the source of the environment and city
  feature shaders) adds the texel through the shared night emissive response.
- `plan.py` holds the catalogs (era rows x C3X building columns, regional
  alternatives, hero buildings alone, city context scenes).
- `page.py` and `content.py` build the recommendation page under
  `Renderer/lab/out/districts/page/`.

```sh
python3 Renderer/lab/studies/districts/night_tree.py prepare   # then build (VM)
python3 Renderer/lab/studies/districts/kits.py compile
python3 Renderer/lab/studies/districts/kits.py render [--catalog campus]
python3 Renderer/lab/studies/districts/kits.py neighborhoods
python3 Renderer/lab/studies/districts/page.py
```

Renders wait while another session's `prlctl exec` runs and resume from
finished frames.

## Findings worth keeping

- Civ VI districts are a base variant per building set x art era (x culture,
  almost never outside City Centers) plus one hero building model per building,
  authored in place in the hex: the same model as C3X's sprite columns.
- 2.5x Civ VI's authored size fills a Civ III tile beside production cities.
- Civ VI places district buildings at hex (60 degree) angles. Each building is
  turned like a city building (`footprint_alignment` from the city pack): footprint
  square to the tile, front toward the nearer of south-west and south-east, about
  its own centre. Long thin pieces (walls, palisades) keep their authored angle.
- Land district kits are re-drafted onto the square tile by role (from Civ VI
  entry names): walls into a square ring with corner towers, heroes and large
  landmarks to the back of a 3 x 3 lattice, fillers (tallest first) in the other
  slots; generic trees and hex ground platforms dropped; drawn at 2.0x.
- Where Civ VI's district kit is thin, Civ VI city buildings fill in by era:
  row houses (`DIS_CTY_INDRH_*`) and glass towers (`DIS_CTY_MG*_Bld_*`) in
  Commercial Hubs. The Colosseum is Civ VI's Colosseum wonder (no district
  building exists), the Stadium in the Modern era; AMER keeps the Tlachtli.
- Constructed wonders live only in `WonderMovie.blp`. `kits.py` reads its
  entries as landmarks and skips the bind-pose check for the animated pieces;
  a `merge` part keeps chosen materials within a radius (shell, seating,
  arena) as one chunked mesh each, dropping scaffolds and stone piles.
- Drafting (kits._draft) packs each tile as one seeded irregular cluster on an
  occupancy grid: heroes back, kit buildings, then copies of kit buildings and
  `filler` parts (the city pack's houses, recovered into `city_buildings.json`)
  up to the layout's `density`, at most four copies of any one piece; trees
  scatter into gaps. Walls ring only real wall kits (6+ pieces). Walls of every
  piece align from side-facing normals above ground (`_wall_alignment`).
  Heroes over 0.2 tile^2 shrink (down to 0.6x).
- Part options: `role` (filler, plant, building), `clip_z` (cut a model, e.g.
  the rig's deck without its derrick), `ramp` + `conform` (take out an
  authored slope and seat every vertex on the drawn ground via the renderer's
  `city/walls/` path: the ski gondola). Cell options: `layout` (density,
  trees, props, reserve, pond), `draft: "force"`, `snow` (Civ III snow-capped
  bit through renderer.py), `snow_cover` (snow decal over a mountain).
- More layout keys: `copies`, `building_scale`, `limit`, `hero_area`,
  `pond_offset`, `walls: "explicit"` (catalog wall kit with `gate`). Part keys
  `squash` (flatten height), `lift`. Cell keys `snap`, numeric `facing`,
  `runway` (decal beside airfield buildings; Civ VI's runway is an unplaceable
  decal and the airport tarmac is alpha-cut). Long thin pieces align on their
  footprint, not wall normals (palisade stakes have none).
- Holy Site "no buildings" is a pond decal (Civ VI water is shader-only) ringed
  by trees. Constructed wonders and the factory stack come from
  `WonderMovie.blp` / `tilebases.blp`.
- District bases have no ground mesh (Civ VI paints the hex with its terrain
  system). Drafted districts get era paving (`CityAccentsLab/ground`) only
  round each standing building: `_footprints` boxes the placed outline of
  each mesh above ground (z > .003; Civ VI foundation skirts are wider), the
  paving field adds a .01 margin and short MST paths (< .12 tile), no plaza;
  it is cut at its zero contour (the composition path alpha-tests) and drawn
  as a `decal/` so it follows the terrain.
- A BC3 colour texture's alpha is Civ VI's tint mask (panels recoloured per
  building; on the DIS_NBH houses, the roof planks), not a cutout. A part's
  `tint` option `(roof, wall)` multiplies the masked texels by the first
  colour, paints pale unsaturated unmasked texels (white weatherboard) with
  the second and tones down saturated orange brick (`_tinted`, BC1 out; only
  masked materials get a tinted copy). Neighborhood houses use plan.HOUSE_TINTS.
- The shared night line passes emission only for city, mine and infrastructure
  weights; the night tree adds flagged resource bodies to that mask.
- Layouts are templated per `layout_key` (one per column), so a column keeps
  its layout across eras; `fixed` parts keep their size.
- The shared compound importer rejects district bases (trailing empty model),
  some attachment policies and the Arena's skin weights. `kits.py` patches copies
  of those functions in-process; the shared importer is unchanged.
- Gathering Storm's `BUILDING_POWER_PLANT` is the nuclear plant. Poland's
  Sukiennice package fails a size check in this install. The Great Wall kit
  (`Walls.artdef` GreatWallSet) imports through the city-wall path and is laid
  out with C3X's tile topology (pillar per tile, half walls to edge neighbours).
