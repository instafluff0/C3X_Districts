# Farms

Lab-only until the user accepts it; production `farm_runtime.bin` is unchanged
until then. The new behaviour switches on only for a farm pack with a
`farm_kit` group, so game builds from this checkout draw farms as before.

## Farm kit

Production since 2026-10-06 (user-accepted; `farm_runtime.bin` carries a
`farm_kit` group). A pack without that group keeps the earlier quadrant fields.

- **One green patchwork everywhere.** The planted crop atlas is Civ VI's whole
  farm patchwork; its ~30 fields become separate convex pieces sharing one
  placement per farm tile: any angle, the authored size (`--field-size`, 3.3
  tiles) up to a fifth larger, shifted so each tile shows a different part.
  The tan and muddy palettes are not used; the ground shows between fields.
- **Open ground.** Pieces are clipped to their tile, the drawn route
  centerlines, the resource's parts (`resource_footprint`, .05 rounded margin,
  .04 more for animated herds) and water. Route verges taper from .095 (road)
  / .12 (railroad) on a tile with one or two route lines to .045 / .06 at a
  dense junction. Three or more route lines select the gap-free patchwork
  (`farm_kit:dense`, the opaque atlas with every path given to its nearest
  field): the routes become the field edges. Trees and the farmhouse keep the
  same clearance.
- **Organic edges.** Field decals carry their cut fade (0 at a tile, route,
  yard or water cut, 1 from .04 inside) in their material's spare digits
  (.0131-.0134; kit props keep .0135). `terrain_scene.hlsl` blends them with
  their source fringe instead of a hard alpha cutoff, and `source_caster.hlsl`
  keeps them out of the shadow map. Farms draw before the route strips (which
  write no depth), so a road or railroad always paints over the fields.
- **Slivers.** A field the tile edge cuts to a strip narrower than about .07
  tile (fringe included) or a small scrap is dropped whole, as is one left as
  crumbs (< .004 tile²). Fields that routes divide stay.
- **Draped and sorted on the ground.** Fields drape on the rendered natural
  ground a hair below route strips. Kit pieces carry material fraction +.0035
  (.0135/.0235/.0335), which `world_projection.hlsl` and `rigid_feature.hlsl`
  give the natural height-depth basis; the feature basis hid farms on low
  relief and hills. `Renderer/tools/overlay_farm_shading.py` carries that into
  the pinned Renderer64ResidentRuntime pack (`farm-overlay.json`).
- **Resource kits.** `farm_kit:<name>` swaps the patchwork for a resource
  (ripe wheat by default: about 80% of its fields recoloured to straw from BC1
  endpoints only); `farm_kit:<name>:crop` also drops the yard.
- Irrigated tiles keep their resource facts in object identity
  (`CapturedScene::object_inputs`), so a farm rebuilds when its resource changes.

## Plots

Production since 2026-10-07 (user-accepted; built with `--plots`, previous
pack kept as `farm_runtime-before-plots-20261007.bin`).
`build_farm_runtime.py --plots` adds `farm_kit:plots` (and
`farm_kit:<resource>:plots`): ten rectangular fields of the planted atlas
that replace the patchwork. `lay_out_farm_plots` labels the open ground that
routes, the yard and water split apart (`FarmClearing::regions`, 32x32 cells).
Each region lays strips along the route it borders most, from its verge
outwards and evenly over the region's depth (open regions: the shore's
tangent, else a world lattice on the 3x3 map region's axis). Strips along a
route divide end to end into whole plots, stretched up to a third
(`Instance::stretch`) to fill their place. Each plot is clipped to its own
region (`Instance::region`), so plots follow the routes' contours and never
straddle one.

Joined areas: Civ III links every two neighbouring tiles that carry a route,
so the network's areas are unions of the triangles that split each square of
four tile centres along its diagonals. Each farm reads its neighbours'
route and irrigation bits (`settle_farm_fields`' `near` lookup, recorded as
dependencies like the routes') and lays out an area of up to 32 triangles
whole: strips along its longest straight route, the same in every tile the
area touches, so its plots run across tile edges. A joined plot (region
+0x100) is cut exactly at an edge shared with another farm
(`FarmClearing::shared`), without feather or sliver checks there; larger
areas keep the per-tile layout above. Still to confirm in game: a farm re-lays
its plots when a neighbour gains a road or irrigation.

Cost: plot meshes use about the patchwork's .078-tile cells (five rows), and
the water clearance is sampled on a 9x9 lattice. With those, the 1498 AD
`near` capture matched the patchwork: required GPU bytes within 0.4%, and 1x
scroll, minimap jumps and 2x/3x scroll within run noise. A 12x12 grid with
per-cell water queries cost +9% GPU bytes and slowed scrolling and jumps.

## Lab

```sh
python3 Renderer/tools/asset_compiler/build_farm_runtime.py --output "farm_runtime~candidate.bin"
python3 Renderer/lab/studies/farms/study.py render before
python3 Renderer/lab/studies/farms/study.py render after --farm-runtime "farm_runtime~candidate.bin"
$C3X_RENDERER_PYTHON Renderer/lab/studies/farms/study.py sheet before after
```

Builder options: `--field-size`, `--solid` (grassy lanes instead of the
ground between fields), `--ripe Wheat` or `--ripe Wheat:crop`.

Balance options (user-accepted and promoted 2026-10-07, all six; staged
12:19, previous pack kept as `farm_runtime-before-balance-20261007.bin`; on
the 1498 AD `near` capture required GPU bytes rose 2.5% and timing stayed
within run noise; plots packs only, each a
pack group or texture a pack without them lacks). Production builds with
`--plots --underlay --green --ditches --narrow --sparse`:
- `--underlay [R,G,B]`: the terrain's own grassland albedo (same world
  scale and alignment, so as crisp as the ground around it; mean colour
  moved to R,G,B, detail contrast x1.8, about 93% opaque) as a decal under
  each farm (`farm_kit:ground`, slot 3). It spans route verges, joins
  neighbouring farms unfeathered and eases into other land and water over a
  wide irregular edge (feather .2, up to .07 inset); with it, fields and
  props stop up to .04 short of other land, irregularly;
- `--green`: cooler, richer crop greens whose rows calm toward each field's
  mean on smaller mips (`CALM`), so farther zooms read calmer;
- `--ditches`: sparse irrigation ditches, .02 wide, light water (slot 2),
  between strips and beside routes (`farm_kit:ditch`). Dropped from
  production 2026-10-07: the user found the blue lines out of place;
- `--narrow`: narrower route verges in farmland (`farm_kit:narrow`);
- `--sparse`: 2-3 trees (about 2 kept) and a farmhouse on about three farms
  in four (`farm_kit:sparse`; the user's in-between density, 2026-10-07).
Terrain identity (user-accepted and promoted 2026-10-07, staged 16:40, with
the game shader overlay `farm-overlay.json`; 1498 AD required GPU bytes about
unchanged; production builds with
`--plots --underlay --green --narrow --sparse`; rollback copies in
`rollback-before-terrain-20261007/`):
- The farm ground's strength follows its tile's terrain: grassland and flood
  plain 1, plains .92, tundra .88, desert .45 (sand shows between fields).
  It varies gently in world space, and two farms of different terrains meet
  halfway at their shared edge.
- Its tint t (tundra 0, grassland .25, flood plain .4, plains .65, desert 1)
  goes on the ground fully and on the planted crops at .7 (.35 on desert).
  `farm_kit_ground_tint` in `terrain_scene.hlsl` recolours each channel.
- t rides in the decal normal's length (1.1 + t; plain decals 1). The compact
  48-byte feature vertex (`prepared_mesh.h`) keeps the normal but drops
  `world_valid`, and the shader normalizes the normal before any lighting.
- `farm_kit:plots:ripe` ripens about 5% of plots (12% on plains, 2% on
  tundra), by the terrain under each plot's centre.
- The ground feathers over .32 tile at other land, up to .1 inset; fields and
  props there feather over a band 2.5 times wider.
- Lab packs pass `--tag NAME` so their derived textures are written as
  `*~NAME.dds`. Production rebuilds overwrite the untagged names, so swap
  them only at stage time.

Derived textures are named by option (`patchwork_green.dds`,
`patchwork_ripe~green.dds`, `ground.dds`, `ditch.dds`); production uses
these names, so a Lab experiment that changes their parameters must write
other names first. Lab files sit
beside the production file (`C3X_RENDERER_FARM_RUNTIME` selects one). Cases
live in `cases.py`: terrain patches, routes, resources, water and a dense
late-game network.
Regression tests: `Renderer/native/test_farm_kit.py`. `test_biq.py` still renders
the production farms over the unchanged `test.biq` terrain.
