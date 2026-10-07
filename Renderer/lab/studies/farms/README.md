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

## Lab

```sh
python3 Renderer/tools/asset_compiler/build_farm_runtime.py --output "farm_runtime~candidate.bin"
python3 Renderer/lab/studies/farms/study.py render before
python3 Renderer/lab/studies/farms/study.py render after --farm-runtime "farm_runtime~candidate.bin"
$C3X_RENDERER_PYTHON Renderer/lab/studies/farms/study.py sheet before after
```

Builder options: `--field-size`, `--solid` (grassy lanes instead of the
ground between fields), `--ripe Wheat` or `--ripe Wheat:crop`. Lab files sit
beside the production file (`C3X_RENDERER_FARM_RUNTIME` selects one). Cases
live in `cases.py`: terrain patches, routes, resources, water and a dense
late-game network.
Regression tests: `Renderer/native/test_farm_kit.py`. `test_biq.py` still renders
the production farms over the unchanged `test.biq` terrain.
