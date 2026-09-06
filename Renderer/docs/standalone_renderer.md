# C3X Standalone Whole-Viewport Renderer

## Implemented M4.1 Boundary

`standalone/whole_viewport_renderer.py` is the source-independent reference implementation for the first whole-scene renderer milestone. It consumes three previously gated contracts:

1. A validated `c3x.visible_scene.v0` scene.
2. A merged `c3x.renderer_definition_catalog.v0` catalog.
3. Normalized `c3x.asset_pack.v0` packs addressed by stable logical asset IDs.

It owns no window and presents no swap chain. The output is an in-memory viewport-sized color target plus depth, instance-owner, and primitive-owner buffers. The command-line entry point writes that target to a deterministic PNG for inspection.

## Projection And Depth

Every mesh vertex is expressed in normalized tile units and projected from its captured item anchor using the scene's exact pixel basis:

```text
screen = captured_anchor
       + local_x * tile_x_basis_px
       + local_y * tile_y_basis_px
       + local_z * elevation_basis_px
```

Pack preview dimensions never replace the captured basis. The renderer clips all writes to `viewport.map_rect_px`, leaving the retained Civ III HUD/UI region untouched.

The orthographic depth coordinate is `world_x + world_y + 2 * world_z`; larger values are nearer. Depth is barycentrically interpolated per pixel. Equal-depth shared terrain edges retain the first deterministic write, while nearer relief rejects later hidden fragments.

## Lighting And Initial Seasonal Response

The scene supplies the authoritative C3X hour and season. The named `#Environment` supplies sun azimuth, noon color, midnight ambient color, night exposure, and the seasonal-material switch.

- Day strength follows a clamped cosine: zero at night and one at noon.
- Sun elevation reaches 62 degrees at noon; the authored azimuth remains fixed for this initial implementation.
- A per-channel ambient term blends from the configured midnight color/exposure to a documented daylight ambient.
- Diffuse lighting uses interpolated normalized mesh normals and a Lambert term.
- When `seasonal_materials = true`, deterministic renderer-level tints provide the first response: summer `(1.04, 1.00, 0.90)`, fall `(1.10, 0.88, 0.68)`, winter `(0.80, 0.90, 1.10)`, and spring `(0.92, 1.07, 0.94)`.

These tints are deliberately small and generic. Pack-authored material variants and richer environment transitions remain later production work.

## Pack Lookup

Normalized manifests now expose a general logical-ID table:

```json
"assets": {
  "terrain/grassland/base": {
    "type": "terrain",
    "mesh": "meshes/flat_terrain_patch.json",
    "material": "materials/grassland.json"
  }
}
```

All payload paths are resolved beneath the declared pack root. Missing manifests, IDs, meshes, or materials are removed from the resolver's availability set so affected scene items follow the existing Civ III fallback contract. The loader currently accepts normalized triangle JSON meshes, `c3x.material.v0`, and BC3 DDS base-color textures; this is an explicit first-renderer capability limit, not a source-format dependency.

## Lifecycle And Command

The renderer has explicit `ready` and `closed` states. A changed scene viewport recreates color, depth, owner, and primitive targets; an unchanged viewport reuses the generation. Teardown is idempotent and rendering after teardown is rejected.

After building the local normalized grassland pack, render the recorded scene from the C3X root with:

```powershell
py -m Renderer.standalone.whole_viewport_renderer `
  --scene Renderer\samples\scenes\grassland_viewport.scene.json `
  --default Renderer\samples\config\default.custom_rendering.txt `
  --mod-root . `
  --output Renderer\preview\out\grassland_scene_640x480.png
```

Portable synthetic verification is named `standalone_whole_viewport_renderer`. It covers byte determinism, two viewport sizes, authoritative anchor ownership, map-rectangle bounds, true hidden-fragment depth rejection, environment differences, safe fallback, and renderer recreation/teardown without Civ III.

