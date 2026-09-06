# Dedicated city and unit asset pipelines

The first executable unit vertical slice, its mesh-local skin-palette finding,
and the source-independent Civ III owner-color lookup are documented in
[unit_asset_conversion.md](unit_asset_conversion.md).

Cities and units are separate offline/lab pipelines. They may reuse proven
low-level normalized model, material, skeleton, skin, and animation primitives,
but they do not share a source graph, composition recipe, category namespace,
preview gate, or eventual fallback boundary.

Nothing in this contract enables production or game integration. City L17 and
unit L20 remain blocked behind the preceding Terrain Lab gates and require their
own promotion renders and explicit visual approval. Production ownership remains
blocked behind the complete approved lab ladder.

## City pipeline

The city contract is
`Renderer/tools/asset_compiler/city_pipeline.json`. Its normalized namespace is
`city/` and its composition mode is `generated_compound_city`.

Civ VI `Cities.artdef` does not describe the complete map city. Its checked
entries lead to strategic-view representations. The source-backed map-city
pipeline instead follows:

```text
Cultures.artdef civilization membership
        +
CityGenerators.artdef growth and art-era distributions
        +
GeneratorBlockList culture-tagged CityBuildings entries
        v
normalized compound components
        v
deterministic city composition recipe
        v
civilization/culture × era × Civ III size lab matrix
```

The installed-source inventory finds 134 culture profiles covering 213 named
civilizations, four generator population thresholds, seven art eras, and 2,690
resolved CityBuildings bindings representing 975 unique package/entry assets.
All component package bindings close successfully.

The component stage may reuse the generic compound Landmark/TileBase decoder,
but the city pipeline owns the higher-level footprint, grounding, size, culture,
era, owner color, walls/capital/style composition, retained-label envelope, and
city ambient attachments. Civ III labels, population/production displays, and
HUD remain outside its ownership.

Offline intake is now executable through
`Renderer/tools/asset_compiler/city_asset_importer.py`. The representative local
pack proves all 20 Civ III culture-group/era fallback pools with 44 unique
normalized components, including static emissive material channels and exact
model attachment sockets. The deterministic size, capital, walls, owner-color,
night-light, fallback, and retained-layer decisions are recorded in
[city_rendering_strategy.md](city_rendering_strategy.md). This preparation does
not advance L17 or enable runtime city ownership.

## Unit pipeline

The unit contract is
`Renderer/tools/asset_compiler/unit_pipeline.json`. Its normalized namespace is
`unit/` and its composition mode is `animated_formation`.

The unit resolver starts from the editable 93-record Civ III mapping and follows
only typed unit graph roots: units, formations, layouts, members, movement,
member/unit combat, attacks, combat formations, domains, tint profiles, and
attachment bins. External VFX references remain typed boundaries for the later
effects pipeline instead of being mistaken for unit bodies.

The installed-source inventory resolves all 91 model-backed mappings across 77
unique ArtDef targets. Those closures visit 1,315 nodes and resolve 1,428 typed
internal edges with no unavailable target or unresolved graph. The independent
member-bin intake records 3,007 component terminals with no missing logical BLP
package. Tactical Nuke and ICBM remain the two intentional effect-only fallbacks.

The unit pipeline alone owns member selection, skin/skeleton compatibility,
clips and action bindings, formation layout, eight-direction presentation,
owner color, stacking, and the action turntable. Civ III retains the selection
cursor, health bar, activity marks, stack indicators, and HUD.

Offline family conversion now extends the complete Warrior vertical slice with
Archer, Swordsman, Infantry, and Fighter. It emits 18 normalized components,
32 unique generic clips, and 36 logical action bindings. The small first action
contract covers idle, fidget, move, fortify, attack, defend, victory, and death;
Fighter also retains takeoff, landing, and turns. The
proof deliberately spans ranged/melee/firearm humanoids and aircraft; mounted,
crewed siege, armored-with-crew, and multi-mesh naval compositions remain
separate declared work rather than being flattened. See
`unit_asset_conversion.md` for weight normalization evidence, inferred socket
boundaries, pose-versus-motion conversion, native timing, attack-slot aliasing,
facing/rotation policy, reproduction, and the static/animation
contact sheet. This is offline L20 preparation only.

Owner color remains dynamic. Unit packs carry one neutral material plus a
civ-color weight and the checked `unit_owner_color_runtime.json` contract, not
32 precolored variants. At runtime one 64-by-32 LUT is copied from Civ III's
effective loaded palettes, and each displayed unit selects a row through the
native viewer-conditioned civilization and its `Leader.Color_Table_ID`. This
handles scenario overrides, capture, alternate colors, barbarians, and
hidden-nationality presentation without reconverting unit art.

## Closure and reproduction

`dedicated_object_pipelines.py` validates both contracts, refuses namespace or
composition collisions, inventories the installed ArtDefs and packages, and
emits an ignored evidence report at
`Renderer/preview/out/object_pipelines/city_unit_inventory.json`.

```bash
python3 Renderer/tools/asset_compiler/dedicated_object_pipelines.py --require-closed
python3 Renderer/tools/asset_compiler/city_asset_importer.py
python3 Renderer/tools/asset_compiler/unit_family_asset_importer.py
PYTHONPATH=. python3 -m unittest Renderer.tools.asset_compiler.test_dedicated_object_pipelines -v
```

The strict gate fails on unresolved city component bindings, unavailable mapped
unit targets, unresolved typed unit graphs, or missing unit-bin packages. It
does not claim that all 975 city components or 3,007 unit-bin terminals have
already been converted. The city importer currently converts a representative
proof subset; complete composition and visual acceptance remain L17 work.
