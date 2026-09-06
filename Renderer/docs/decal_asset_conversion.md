# Generic decal asset conversion

`Renderer/tools/asset_compiler/generic_decal_compiler.py` is the lab-side source
adapter for terrain-surface decals. It converts reflected source descriptors and
standalone texture payloads into a source-independent `c3x.decal.v0` contract.
It does not add a production renderer or change game code.

The importer is mapping-driven. `decal_sets.json` supplies source asset names,
ArtDef sets, and stable C3X IDs; adding a proven decal family requires a mapping
entry rather than a new importer path. The current mapping covers five desert
sand decals, nine marsh decals, thirty-one ocean/coast/grassland/plains/hill
surface decals, and sixteen Iron, Coal, Aluminum, Uranium, Diamond, Gold, and
Oil resource decal entries.

## Admission profile

An asset is accepted only when all of these checks pass:

- exactly one reflected `DecalDesc2` array is reachable from its landmark record;
- its terrain-edit vector is empty;
- no conventional model, mesh, primitive-group, or material container is
  attached;
- its descriptor is 108 bytes and contains finite, ordered footprint/content
  bounds;
- base-color and height slots have the expected reflected texture classes;
- optional specular and fog-color slots are exported only when their classes
  match the requested role; and
- every mapped asset has exactly one matching ArtDef placement record.

This intentionally excludes projected meshes and terrain deformation. Those
need separate normalized contracts instead of hidden exceptions in the decal
loader.

## Runtime contract

Each `c3x.decal.v0` document contains a stable asset ID, bounds normalized to
tile units, a full UV rectangle, generic texture channels, clamp addressing,
and terrain-surface render intent. The pack manifest groups variants with
normalized placement controls. A source entry with multiple descriptors becomes
one `compound_decal` manifest asset containing ordered `c3x.decal.v0` parts;
Oil exercises this path with two parts per authored entry. Runtime JSON contains neither source-format
names nor absolute paths.

The source report is written outside the pack to
`Renderer/preview/out/decals/decal_build.json`. It records source paths, hashes,
pointer-chain evidence, reflected texture classes, rejected optional slots, and
the ArtDef names used during conversion. Both the generated pack and report are
ignored local artifacts because derived source art is not redistributable.

## Lab command

From the project root:

```sh
python3 Renderer/tools/asset_compiler/generic_decal_compiler.py
```

The default output is `Renderer/packs/DecalsNormalized`. Alternate installed
asset roots, mappings, pack directories, and report paths can be supplied with
the corresponding command-line options.
