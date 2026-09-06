# Generic terrain-element asset conversion

`terrain_relief_builder.py` now discovers terrain-element package stripe data
from the package's own reflected entry catalog. It no longer requires a
Base-only bootstrap entry, so the same structural reader accepts Base and
Expansion 2 `TerrainElementSet_Base` packages.

The reader treats each 144-byte `TerrainElementPackageEntry` as authority. It
validates the entry-name hash, grid dimensions, finite calibration parameters,
positive height scale, and the typed blob indices for height, blend, region-ID,
and noise channels. Present channels must contain a complete two-level LOD pair
at the dimensions implied by the element grid. Missing optional channels remain
explicitly absent rather than receiving fabricated textures.

`generic_terrain_element_compiler.py` applies a source mapping and writes
source-independent `c3x.terrain_element.v0` documents and tightly bounded R8
DDS resources. Continuous channels use `R8_UNORM`; discrete region IDs use
`R8_UINT`. Source entry names, blob names, installed paths, and hashes remain in
the ignored external build report.

The initial Expansion 2 mapping covers its complete seven-entry package: one
generic volcano feature and six staged natural-wonder terrain elements. These
are normalized source assets only. They do not enable renderer ownership or
advance natural-wonder integration ahead of its milestone.

From the project root:

```sh
python3 Renderer/tools/asset_compiler/generic_terrain_element_compiler.py
```

The default generated pack is `Renderer/packs/TerrainElementsNormalized`; its
source report is `Renderer/preview/out/terrain_elements/expansion2_build.json`.
