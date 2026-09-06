# Route-style conversion

`route_style_importer.py` converts source route descriptions and embedded route
decal materials into reusable `c3x.route_style.v0` assets. It is an offline
Terrain Lab adapter; it does not transfer roads or railroads into production
renderer ownership.

## Normalized contract

A route style consumes a connected centerline graph. The eventual lab route
builder remains responsible for deriving that graph from authoritative tile
connectivity and for conforming it to accepted terrain relief. The style
supplies:

- nominal width in tile units;
- normal and pillaged states;
- tiled-path, endpoint/fadeout, and transition recipes;
- ordered surface layers and their height offsets;
- normalized atlas endpoints for each route piece; and
- source-independent base-color, height, optional specular, and fog materials.

Junctions are composed from incident path branches. This avoids baking one
source game's six-neighbor topology into the pack and keeps the same style
usable with Civ III's authoritative connection graph.

Ordered ArtDef merging and multi-package material resolution are part of the
adapter. This is required for the railroad style: its ballast and rail layers
come from Expansion 2 while its lowest fade/transition layer inherits a base
road piece. Atlas endpoint direction is preserved, including reversed fade and
transition regions.

Bridge and other transition doodad bodies are intentionally not flattened into
the route style. `route_doodad_importer.py` compiles them into the companion
`c3x.asset_pack.v0` pack `RouteDoodadsNormalized`, while
`route_transitions.json` preserves their source route-stage pairs, normalized
lengths, contouring policy, and gap-scaling policy. Both runtime packs contain
only generic C3X documents, meshes, and DDS payloads; neither retains an
installed-source path or source-format dependency.

## Initial proof set

The default mapping compiles:

- ancient, medieval, industrial, and modern road styles; and
- the Expansion 2 railroad style, including ballast and rail layers.

All five styles include normal and pillaged recipes for tiled paths, fadeouts,
and transitions. The generated pack uses ten route materials and 25 ordinary
DDS textures extracted from validated embedded texture ranges.

The companion doodad mapping compiles medieval, industrial, modern, and
railroad bridge bodies. Each body contains separate worked and pillaged mesh and
material bindings. The initial proof set contains four bodies, eight geometry
parts, eight materials, twelve endpoint decals, thirteen unique textures, and
all fifteen bridge transition records declared by the Base and Expansion 2
route ArtDefs.

The Expansion 2 railroad package has a canonical source quirk: its internal
header declares 220,672 bytes while the installed Steam-depot file is 200,192
bytes. The local file's SHA-1 matches the depot manifest. The route mapping opts
into this one package exception explicitly; ordinary CIVBLP readers remain
strict, and the importer still validates every package and big-data range
against the actual file length.

Run from the project root:

```sh
python3 Renderer/tools/asset_compiler/route_style_importer.py
python3 Renderer/tools/asset_compiler/route_doodad_importer.py
python3 Renderer/preview/render_route_doodad_sheet.py \
  --manifest Renderer/packs/RouteDoodadsNormalized/manifest.json \
  --output Renderer/preview/out/route_doodads/bridge_contact_sheet.png \
  --report Renderer/preview/out/route_doodads/preview.json
```

The ignored local packs are written to `Renderer/packs/RouteStylesNormalized`
and `Renderer/packs/RouteDoodadsNormalized`. Installed paths, package offsets,
source identifiers, and hashes remain outside the packs in the corresponding
`Renderer/preview/out/route_styles/build.json` and
`Renderer/preview/out/route_doodads/build.json` reports. The source-independent
contact sheet puts worked bodies on the top row and pillaged bodies on the
bottom row.

This intake prepares art and metadata for L14/L15; it does not implement the
route graph builder, terrain conformance, bridge placement, route rendering, or
production ownership.
