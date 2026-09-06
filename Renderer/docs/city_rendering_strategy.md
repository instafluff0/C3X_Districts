# City intake and rendering strategy

## Result

The offline city intake is executable without starting L17 or production city
ownership. `city_asset_importer.py` resolves the complete generated-city source
graph, selects representative components for the Civ III-facing culture/era
matrix, converts geometry, materials, decals, skeletons, emissive maps, and
attachment sockets, and emits a source-independent local pack.

The current proof pack covers all 20 culture/era fallback pools with 44 unique
components. It contains 75 normalized geometry parts, 127 materials, 96
emissive material bindings, 101 unique textures, and 35 exact attachment
sockets. The paired preview renders all 20 pools in both day and night. This is
an intake proof, not the L17 composed-city render or approval gate.

A second offline pack now converts 19 source-backed wall pieces: five ancient,
seven medieval, and seven Renaissance/industrial pieces, with half-wall,
segment, gate, and tower roles complete in every kit. Those pieces contain 45
normalized geometry parts, 41 materials, and 13 textures. Their topology is
prepared for Lab; it is not yet mapped to a rendered city perimeter.

## Why cities are compositions

The source map city is not one model. `Cultures.artdef` establishes
civilization membership, while `CityGenerators*.artdef` supplies art-era and
growth distributions and its `GeneratorBlockList` records choose tagged
`CityBuildings` components. The installed graph contains 2,690 resolved
bindings representing 975 unique package/entry components. `Cities.artdef`
contains strategic-view assets and is not used as a substitute for the map-city
graph.

The C3X runtime contract therefore uses a generic component pool plus a
deterministic composition recipe. It does not retain source package paths,
source formats, culture tags, or component names.

## Civ III selection matrix

The checked source-to-lab fallback mapping is:

| Civ III culture group | Ancient | Medieval | Industrial | Modern |
|---|---|---|---|---|
| American | AncientWood | SouthAmerican | Colonial | ModernGlass |
| European | AncientEarth | DEFAULT | RowHouse | ModernGlass |
| Mediterranean | DEFAULT | Mediterranean | Colonial | ModernGlass |
| Middle Eastern | AncientEarth | Mughal | RowHouse | ModernGlass |
| Asian | AncientWood | EastAsian | RowHouse | ModernGlass |

These source labels exist only in the offline strategy and build report. The
runtime catalog exposes generic `style × era` pool IDs. A future explicit
civilization override may replace any pool, but the five culture groups remain
the complete fallback so every Civ III city is renderable.

Population is a composition axis rather than another model family:

| Civ III size | Population | Components | Footprint radius | Height scale |
|---|---:|---:|---:|---:|
| Town | 1–6 | 4 | 0.31 tile | 0.92 |
| City | 7–12 | 7 | 0.39 tile | 1.00 |
| Metropolis | 13+ | 11 | 0.46 tile | 1.08 |

Composition is seeded by world seed, stable city ID, and map position. A pool
is exhausted before a component repeats, and slot angles remain stable across
population transitions. Growth should add outer slots rather than reshuffling
the whole skyline. Owner color applies only to authored tint channels or
banners; the complete city is never player-color graded.

## Capitals, walls, and retained information

A capital is an additive center-slot accent, not a replacement city family.
The generic source `BUILDING_PALACE` record names `DIS_CTY_Palace_CP`, but that
string is composition metadata rather than a terminal body. A broader installed
content search subsequently found two culture-specific palace compounds. Both
now normalize in `FutureGateCandidates`, including emissives and exact
fire/smoke socket transforms. L17 may compare them as optional centerpieces
only through an explicit city-style mapping; it must not treat either source
culture as the universal capital. Civ III's native capital icon remains
retained unless the owning gate separately transfers it.

Walls are a separate perimeter kit fitted to the composed footprint. They do
not select a different underlying city. The three converted kits expose
multiple straight/half choices plus gates and towers; L17 must still establish
their perimeter topology and approve both zooms before native walls can be
suppressed. Airport, harbor, barracks, spy-agency, and
embassy indicators likewise remain native initially.

City labels, population and production displays, status icons, and HUD remain
Civ III-owned throughout the first city renderer promotion. L17 may suppress a
native city body only after its complete replacement matrix passes the lab gate.

## Night lights and ambient effects

The component decoder confirms the `Generic_Emissive` material slot and emits
its texture as a generic emissive channel with night activation and a
non-emissive missing policy. C3X's shared environment remains the only clock;
there are no duplicated day/night city assets. The proof matrix demonstrates
emissive response in all 20 fallback cells.

`AttachmentPointList` records also resolve to typed
`AttachmentPointCookData`. Each record names exactly one skeleton bone, so its
normalized socket and local rest transform are now confirmed. The current proof
pack preserves 28 smoke, five flame, one night-light, and one unresolved
semantic socket. A `PIL` name marks a pillaged-state hint.

The attachment resource identity and analytic-light/VFX parameters are not yet
decoded. Every socket therefore carries `binding_status: resource_unresolved`;
the runtime must not invent source-equivalent color, radius, falloff, particle
script, or state behavior. Static emissive windows belong to L17. Animated
flame, smoke, steam, and flicker remain M7.5-owned after generic resources are
resolved or explicitly authored.

## Reproduction and later gates

From the project root:

```bash
python3 Renderer/tools/asset_compiler/city_asset_importer.py
python3 Renderer/tools/asset_compiler/city_adjunct_asset_importer.py
python3 Renderer/preview/render_city_day_night_sheet.py \
  --manifest Renderer/packs/CityComponentsNormalized/manifest.json \
  --output Renderer/preview/out/cities/day_night_matrix.png \
  --report Renderer/preview/out/cities/day_night_matrix.json
PYTHONPATH=. python3 -m unittest \
  Renderer.tools.asset_compiler.test_city_asset_importer \
  Renderer.tools.asset_compiler.test_city_adjunct_asset_importer \
  Renderer.preview.test_render_city_day_night_sheet
```

The normalized pack and rendered evidence are local ignored derivatives and are
not redistributed. L17 still owns multi-component city composition, grounding,
collision/depth behavior, both-zoom readability, the full culture/era/size
matrix, wall composition, a separately authored/resolved capital accent if one
is desired, and its 192-tile promotion render. I17 remains
blocked until that handoff is frozen and approved. No Civ III patch symbol is
needed for this offline work.
