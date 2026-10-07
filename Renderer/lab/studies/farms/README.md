# Farms

Lab-only until the user accepts it; production `farm_runtime.bin` is unchanged
until then. The new behaviour switches on only for a farm pack with a
`farm_kit` group, so game builds from this checkout draw farms as before.

## Farm kit

- **One green patchwork everywhere.** The planted crop atlas is Civ VI's whole
  farm patchwork (about thirty fields of mixed shapes and row directions, soil
  fringes, filled colour under its transparent paths). The kit draws it once
  per farm tile at any angle, at its authored size in tiles (`--field-size`,
  up to a fifth larger at runtime) and shifted so each tile shows a different
  part. The tan and muddy palettes are no longer used.
- **Open ground.** The patchwork is clipped to its tile (a .02 verge), its
  drawn road and railroad centerlines (.075 / .10 half widths), its resource's
  parts (`resource_footprint`, .05 rounded margin) and water (shore and river
  bank). Trees and the farmhouse keep the same clearance.
- **Draped and sorted on the ground.** Fields drape on the rendered natural
  ground (low relief, hills), a hair below route strips. Object meshes sort on
  a feature depth basis that hid farms on raised ground; kit pieces carry
  material fraction +.0035 (.0135/.0235/.0335), which `world_projection.hlsl`
  and `rigid_feature.hlsl` give the natural basis, as for resources and bridges.
- **Resource kits.** `farm_kit:<name>` swaps the patchwork for a resource
  (e.g. ripe wheat, about 80% of its fields recoloured to straw from BC1
  endpoints only); `farm_kit:<name>:crop` also drops the yard, so the farm is
  the resource's own planting.
- Irrigated tiles keep their resource facts in object identity
  (`CapturedScene::object_inputs`), so a farm rebuilds when its resource changes.

## Lab

```sh
python3 Renderer/tools/asset_compiler/build_farm_runtime.py --output "farm_runtime~kit.bin"
python3 Renderer/lab/studies/farms/study.py render before
python3 Renderer/lab/studies/farms/study.py render after --farm-runtime "farm_runtime~kit.bin"
$C3X_RENDERER_PYTHON Renderer/lab/studies/farms/study.py sheet before after
```

Builder options: `--field-size`, `--solid` (grassy lanes instead of the
ground between fields), `--ripe Wheat` or `--ripe Wheat:crop`. Lab files sit
beside the production file (`C3X_RENDERER_FARM_RUNTIME` selects one). Cases
live in `cases.py`: terrain patches, routes, resources and water.
Regression tests: `Renderer/native/test_farm_kit.py`. `test_biq.py` still renders
the production farms over the unchanged `test.biq` terrain.
