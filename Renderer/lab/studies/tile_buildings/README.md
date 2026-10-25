# Tile buildings gallery (Lab)

Civ VI stand-ins for the tile buildings Civ III draws in
`Map_Renderer::m12_Draw_Tile_Buildings`: fortress, barricade, airfield, outpost,
radar tower, colony and scenario victory point. Under the custom renderer the
native tile plane is skipped, so before 2026-10-23 none of these was drawn.
The user's picks are integrated: `sitepack.py` freezes them into
`Renderer/packs/TileBuildingsFrozen`, which the production site pack merges
(`Renderer/lab/categories/objects/huts-camps/README.md`).

- `plan.py`: catalogs, Civ III eras by option. Colonies rebuild the
  2026-10-07 draft's concepts (`Renderer/lab/out/colonies_survey`) with an
  explicit back arc, leaving the tile's middle and front to the resource.
- `study.py`: drives the district survey's compile/bake/render pipeline
  (`studies/districts/kits.py`, unchanged) with this study's Lab packs
  (`TileBuildingKitSourcesLab`, `TileBuildingKitCatalogLab`) and a pinned
  DLL + preview pair (`lab/out/tile_buildings/pinned`). `refs` cuts Civ III's
  sprites from TerrainBuildings.PCX, x_airfields and detect.PCX and x_victory.pcx.
- `page.py`: writes the self-contained `lab/out/tile_buildings/gallery.html`.
- `sitepack.py`: bakes `plan.picks` (named as the runtime looks them up) into
  the frozen pack; `--lab-pack TileSitesLab` also builds a Lab site pack.
  The Ancient barricade ring has no gate: Civ VI's Ancient gate is an open
  gateway, which read as a breach.
- `study.py scene`: the 1498 AD save's Japanese barricade line, with every
  family placed nearby, rendered through the production site path.

```sh
python3 Renderer/lab/studies/tile_buildings/study.py refs
python3 Renderer/lab/studies/tile_buildings/study.py compile
python3 Renderer/lab/studies/tile_buildings/study.py render [--catalog tb-colony]
python3 Renderer/lab/studies/tile_buildings/study.py crops
python3 Renderer/lab/studies/tile_buildings/page.py
python3 Renderer/lab/studies/tile_buildings/sitepack.py --frozen TileBuildingsFrozenLab --lab-pack TileSitesLab
python3 Renderer/lab/studies/tile_buildings/study.py scene [--pack TileSitesLab]
```

Night light (2026-10-25): Civ VI lays a building's emissive map out by uv2
(as the city pack reads it), not by its colour's uv0, which the kit squares it
by. Imports keep uv1/uv2 (`study.py` patches the borrowed compiler), and
`sitepack.light_sources` splits each emissive draw into its dark faces and the
faces uv2 lights, baking their light by uv0 (`build_mine_runtime.bake_light`)
into one texture per colour texture. The picks catalog sets kits' `"night":
"sources"`, so `kits.bake` neither splits nor merges again (its own uv2 split,
for districts, would split these twice). Lit parts are named `:glow`; their glow
atlas, kept at a quarter of the colour atlas's resolution, takes a spare site
pack slot (`:g<slot>`, owner code .181-.188, feature shaders' tile glow).
Night_light points (Civ VI's lantern and light bones) become `lamps/<mesh>`
companion assets, which Renderer64 draws as city lamps where the building
stands (`object_compiler.h tile_building_lamps`). The airstrip lights 7.5% of
its surface (uv0 would light 45%) and has 8 warm lanterns. The aerodrome lights
20% and has 10 white runway lights and 7 mast lights; the mast lights are taken
as red blinkers (`FX_Light_Airport_Tower`'s colour is not decoded). Lamps
show only in game (Renderer64); Lab frames show the glow. Freeze into a Lab
pack with `--frozen TileBuildingsFrozenLab --lab-pack TileSitesLab`; plain
`sitepack.py` writes production's `TileBuildingsFrozen`.

Findings:
- The Alcázar (Vikings Landmarks) names its buildings without a package; the
  borrowed compiler looks in `landmarks/tilebases`, so `study.adopt_children`
  retries them in the root's own package (`IMP_Alcazar`).
- The clutter obelisk has only a `default` state; its cell sets `state`.
- Lone towers and markers need about 2.4x the district scale to stand as tall
  as Civ III's outpost and radar sprites; colonies need 1.8x.
- In Civ III a unit stands in its fortress. The Fort keeps (castle ring, star
  fort, concrete fort) have open courtyards; Lab frames show no unit yet.
