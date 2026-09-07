# City intake and rendering strategy

## Result

The offline city intake is executable without starting L17 or production city
ownership. `city_asset_importer.py` resolves tagged generated-city block bindings,
selects representative components for the Civ III-facing culture/era
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

## Generator and city-ground follow-up

**Current user preference: one era only per city.** Mixed-era source distributions
are reference evidence, not the selected appearance. Use applicable generator
growth, placement and ground data within the current-era family.

The current Lab investigation found that the importer omitted authored generator
and grounding collections. [Recovered metadata and ground-piece evidence](../terrain_lab/v2/audits/beauty/CITY_GENERATOR_FINDINGS.md)
now records mixed-era ordering, weights, population growth/fill parameters and
exact ground-decal triangle UVs. The current city recipe still does not implement
the recovered generator. Its next replacement should use those parameters with
Civ III geometry; ordinary roads remain deferred by user instruction. A combined
modern paving probe exists, but its narrow visible strip is not a new city best.

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
string is composition metadata rather than a terminal body. The original
narrow DLC search found two culture-specific compounds; the complete installed
ArtDef audit now finds 47 distinct standard-game palace roots covering regional,
generic, and civilization-specific styles. All 47 palace bodies normalize into
the separate `CityPalacesNormalized` pack with emissive materials and exact
attachment transforms. The [palace intake](city_palace_asset_import.md) records
the full roster, selection contract, and the Gran Colombian root's four still
unresolved required tree children. L17 may compare these as optional
centerpieces only through an explicit city-style mapping; it must not treat a
source civilization or culture as a universal capital. Civ III's native
capital icon remains retained unless the owning gate separately transfers it.

The user's September 2026 capital request now has a combined Lab comparison:
[r13 palace evidence](../terrain_lab/v2/audits/beauty/CITY_CAPITAL_r13_EVIDENCE.json).
The Mesoamerican palace is an additive body in an explicitly mapped American
ancient-style fixture; seven surrounding house placements remain identical in
the palace-off control. `capital_styles.json` under the Lab objects system owns
the experimental mapping and uniform footprint calibration. This is only one
entry in the broader offline library, not a universal capital mapping. The
Gran Colombian candidate still has four unresolved required tree attachments
and is not claimed as a complete kit.

Production selection must read authoritative captured capital state
(`city_flags & C3X_RENDERER_CITY_CAPITAL`, exported as `is_capital`) and the
resolved city style. It must not infer capital status from population, city
order, building appearance, or a particular civilization. Missing palace art
leaves the ordinary city body and native capital indicator intact. A capital
change invalidates the city accent; ownership and retained UI gates still apply.
This Lab comparison does not enable a native city implementation.

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
python3 Renderer/tools/asset_compiler/palace_asset_probe.py
python3 Renderer/tools/asset_compiler/palace_asset_importer.py
python3 Renderer/preview/render_city_day_night_sheet.py \
  --manifest Renderer/packs/CityComponentsNormalized/manifest.json \
  --output Renderer/preview/out/cities/day_night_matrix.png \
  --report Renderer/preview/out/cities/day_night_matrix.json
PYTHONPATH=. python3 -m unittest \
  Renderer.tools.asset_compiler.test_city_asset_importer \
  Renderer.tools.asset_compiler.test_city_adjunct_asset_importer \
  Renderer.tools.asset_compiler.test_palace_asset_importer \
  Renderer.preview.test_render_city_day_night_sheet
```

The normalized pack and rendered evidence are local ignored derivatives and are
not redistributed. L17 still owns multi-component city composition, grounding,
collision/depth behavior, both-zoom readability, the full culture/era/size
matrix, wall composition, a separately authored/resolved capital accent if one
is desired, selection and calibration of the prepared palace library, and its
192-tile promotion render. I17 remains
blocked until that handoff is frozen and approved. No Civ III patch symbol is
needed for this offline work.
