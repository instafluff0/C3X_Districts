# L8 Forest And Jungle Source Audit

## Production-bound rule

Terrain Lab no longer accepts third-party proxy meshes for an asset-bearing pass. Procedural topology is allowed only when the same generated topology is intended to ship. Forest, jungle, rocks, vegetation, buildings, and other authored bodies must be normalized from the selected upstream source before they enter a lab acceptance image.

The former Kenney Nature Kit vegetation path is rejected and removed. L8 is complete only through the installed Civ VI feature set and the source-agnostic C3X pack contract described below.

## Installed Civ VI source

`Features.artdef` maps `FEATURE_FOREST` to `CLUTTER_FOREST` plus `CLUTTER_FOREST_SNOW`, and maps `FEATURE_JUNGLE` to `CLUTTER_JUNGLE`. All three sets resolve through `Clutter.artdef` to the installed `environment/clutter.blp` package.

The base forest set declares 15 authored components: three individual pine variants, two pine clumps, two boulder clumps, three tree decals, two shrubs, two dirt decals, and one blend record. The snow set declares 16 components. The jungle set declares 18 components: four grasses, three palms, three plants, four boulder variants or clumps, two decals, one iOS fallback plant, and one blend record.

The package is present: 8,318,976 bytes total, with 2,005,504 bytes of package metadata and 1,383 declared big-data entries. It contains the named landmark entries plus reflected `BLP::VertexBufferEntry`, `BLP::IndexBufferEntry`, `BLP::TextureEntry`, and `FGXModel::ContainerDesc` records. Matching installed texture payloads include the tree, pine, deciduous, jungle, dirt-decal, leaf-decal, forest-terrain, and jungle-terrain families.

The package also contains 53 byte ranges with valid Granny/FGX headers. This is not a loose-model escape hatch: loading the exact `Jungle_Palm_01_Anim.gr2` range and a second generic embedded range through the checked-in CivNexus6 dependencies reports `models=0 meshes=0` for both. Those payloads are animation-side containers. The authored static geometry must therefore be resolved through the package's typed model/container and vertex/index buffer records.

## Verified set extraction

`tools/asset_compiler/clutter_blp_extractor.py` now decodes all 22 unique desktop vegetation bodies selected by the three upstream sets: seven base-forest variants, five snow-forest variants, and ten jungle variants. It locates the 16,751-record static allocation table structurally, follows each landmark through its typed user-data and base-model records, binds one model/mesh/primitive/material, and reads the referenced vertex and index buffers from bounded big-data ranges.

Two fail-closed vertex profiles are proven: format `0x315CFCD9` with a 24-byte stride and format `0x6679B170` with a 32-byte stride. Both store XYZ as half floats at byte 0 and UV0 as half floats at byte 8. The adapter recomputes normals from the actual triangle topology, recenters and grounds each source mesh, and applies one shared 12-source-units-per-tile conversion so relative body scale is preserved. It does not guess the package's packed tangent-frame encoding.

Their material records bind the installed pine, snow-pine, or jungle base-color, gloss, and two LEAN normal textures. The adapter converts those standalone `CIVBIG` resources to standard DDS and emits only generic C3X mesh/material/manifest JSON into the ignored `Renderer/packs/VegetationNormalized/` build output. It also normalizes the exact ArtDef scale, count, scale-variation, priority, overlap, decal, rotation, width, center-model, and minimum-count placement fields. The three runtime groups contain 7/10, 5/8, and 10/10 unique variants/placement records for base forest, snow forest, and jungle.

All 33 unique ArtDef asset names are accounted for: 22 vegetation bodies are normalized and 11 entries are explicitly routed or excluded. Rock bodies move to the authored-rock intake, dirt/jungle decals have no static feature `Model` container, and `Jungle_Clump_IOS_01` is the explicit low-end iOS fallback. Duplicate ArtDef placements remain in the generic placement recipes without duplicating mesh payloads.

## Current boundary

The labeled `vegetation_l8_contact_1648x2322.png` acceptance sheet renders two views of every normalized body using the real material atlas, one shared preview zoom, common source-relative scale, and no proxy art. The corrected UV channel keeps palm trunks brown and fronds green. The 824x1161 box-filtered overview remains readable.

L8 is complete. Its durable outputs are:

1. a fail-closed local source adapter covering every declared set member;
2. normalized meshes, material textures, bounds, and generic placement recipes;
3. explicit routing/exclusion evidence for non-vegetation members; and
4. native-size and thumbnail set-level contact sheets.

L9.4 now consumes this exact normalized pack and placement contract in the four-tile beauty scene, with vegetation-only and no-vegetation isolation renders. `L9_4_VEGETATION_AUDIT.md` records its rejected dense draft and final acceptance candidate. Production integration remains after the lab exit gate.

No Civ VI names, paths, BLP/FGX parsing, or installed-source assumptions may enter the runtime renderer.
