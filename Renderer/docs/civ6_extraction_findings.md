# Civ VI Extraction Findings

## Local Asset Layout

The local Civ VI base path used by the tools is:

`Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets\Base`

Important subfolders:

- `ArtDefs/`: large XML-like metadata files such as `Terrains.artdef`, `TerrainStyle.artdef`, `Landmarks.artdef`, and `Clutter.artdef`.
- `Platforms/Windows/BLPs/`: cooked platform packages and many extensionless cooked assets.
- `Assets/`: gameplay/UI/map data, not the primary loose model source.

## Practical Finding

The BLP tree is a useful index target but should be treated as cooked content. The first compiler slice therefore records candidate terrain/mountain names and paths, but does not try to copy or redistribute Firaxis assets.

Grassland proof-of-concept indexing confirmed that `TerrainStyle.artdef` points at `ART_DEF_TERRAIN_MATERIAL_GRASSLAND` in `terrain/TerrainMaterialSet_Base`, and the corresponding `.blp` package contains readable package/type names plus that art entry.

Standalone texture files in `BLPs/SHARED_DATA` use a 48-byte `CIVBIG` header followed by a block-compressed mip chain. The format field is the standard DXGI format number. For example, `TEXTURE_TER_Grass_Decal_B` reports 1024x1024, 9 mips, and DXGI format 78 (`BC3_UNORM_SRGB`); its declared payload size exactly equals the expected BC3 mip sizes. The asset compiler now validates this structure and wraps the payload in a standard DDS DX10 header. DirectXTex can then convert the DDS to PNG without interpreting proprietary package metadata.

This proves standalone texture extraction. The grassland POC still selects the clearly named grass decal texture as its first visible source asset; the material textures resolved below use a different, embedded storage path.

## M1.2 Grassland Material-Binding Probe

`tools/asset_compiler/civblp_probe.py` now performs a bounded, read-only probe of `terrain/TerrainMaterialSet_Base.blp`. It reads the 28-byte file header and the declared 61,952-byte package-data region, then skips the approximately 293 MiB big-data payload. Its deterministic local evidence report is `civ6_grassland_material_probe.json`.

Proven by header validation and typed allocation-pointer traversal:

- The package is `CIVBLP` version 2, declares 79 big-data entries, and has a 275-entry allocation table at file offset `0xCAC2` with 40-byte entries.
- The package-block and temp-data stripe bases resolve to file offsets `0x86EC` and `0xC95B`. These bases are supported respectively by 114 valid typed string allocations and four reflected allocation-type strings.
- Allocation pointer 86 identifies one 128-byte `TerrainMaterialPackageEntry` candidate at `0xB3CC..0xB44C`. Its qword at relative offset `0x38` points to the unique `ART_DEF_TERRAIN_MATERIAL_GRASSLAND` string allocation (pointer 92).
- Four non-null qwords in that material record resolve through typed child allocations into bounded 104-byte `BLP::TextureEntry` elements:
  - relative `0x48`, pointer 88: logical name `TER_Grass_B`, class string `Terrain_BaseColor`;
  - relative `0x50`, pointer 89: logical name `TER_Grass_H`, class string `Terrain_Heightmap`;
  - relative `0x58`, pointer 90: logical name `TER_Grass_G`, class string `Terrain_Spec`;
  - relative `0x60`, pointer 91: logical name `FOW_Ground_Grass_2k`, class string `Terrain_FOWColor`.
- The package's reflection strings include `m_pBaseColorTexture`, `m_pHeightTexture`, `m_pSpecTexture`, `m_pFOWColorTexture`, and `m_pFuzzTexture`.

Still inferred or unknown:

- M1.2 does not map the material-record offsets to the reflected field names. The adjacent logical class strings strongly suggest roles, but M1.3 must prove the field layout instead of treating string content as sufficient.
- No fifth non-null texture pointer was found in this material record. The likely fuzz slot is null, but its exact field offset and interpretation remain unknown until M1.3.
- M1.2 did not yet establish whether the logical texture-entry names referred to standalone `SHARED_DATA` files or embedded resources. In particular, the probe did not hardcode a `TEXTURE_...` grassland filename.
- Other material integers, hashes, inherited fields, and null qwords remain `unknown_*` values in the JSON report.

The probe is intentionally scoped to this observed Civ VI package layout. Its allocation-table marker, direct and `String::BasicT` storage forms, stripe-base inference, typed parent-element resolution, boundary validation, and deterministic output are covered by synthetic tests. It is not a full generic `CIVBLP` decoder.

## M1.3 Material Roles And Embedded Resources

`tools/asset_compiler/civblp_material_resolver.py` extends the typed pointer chain without reading or copying texture payloads. Its deterministic result is `civ6_grassland_material_binding.json`.

Cross-record layout evidence establishes the following:

- All 31 typed `TerrainMaterialPackageEntry` records resolve relative offsets `0x48`, `0x50`, `0x58`, and `0x60` respectively to texture classes `Terrain_BaseColor`, `Terrain_Heightmap`, `Terrain_Spec`, and `Terrain_FOWColor`. Role selection therefore comes from stable field/class consistency rather than grassland filenames.
- The next qword, at `0x68`, is null in all 31 records. Its fuzz meaning remains medium-confidence because it is inferred from the reflected field-name sequence; no live pointer exists to provide typed confirmation.
- The sole 79-element `BLP::TextureEntry` array uses 104-byte records. Across all 79 records, logical-name and class pointers uniquely resolve at `0x08` and `0x40`; format/dimensions/depth/unknown/mip-count metadata uniquely validates at `0x58`; embedded offset and byte count uniquely validate at `0x20` and `0x28`.
- Every one of the 79 embedded byte counts exactly matches the block-compressed mip-chain size implied by its DXGI format, dimensions, and mip count, and every range is bounded by the package big-data region.

The grassland binding is:

| Role | Material offset | Logical name | Texture class | Format / color space | Storage |
| --- | ---: | --- | --- | --- | --- |
| Base color | `0x48` | `TER_Grass_B` | `Terrain_BaseColor` | BC3 sRGB | Embedded CIVBLP big data |
| Height | `0x50` | `TER_Grass_H` | `Terrain_Heightmap` | BC4 linear | Embedded CIVBLP big data |
| Specular | `0x58` | `TER_Grass_G` | `Terrain_Spec` | BC4 linear | Embedded CIVBLP big data |
| FOW color | `0x60` | `FOW_Ground_Grass_2k` | `Terrain_FOWColor` | BC1 sRGB | Embedded CIVBLP big data |
| Fuzz | `0x68` (inferred) | null | null | n/a | n/a |

This corrects the earlier M1.3 premise that these logical names must resolve to matching standalone `SHARED_DATA` resources: those files are not present, and the texture records already provide a deterministic embedded-resource rule. Standalone `CIVBIG` extraction remains valid for assets that actually use that storage form. M1.3 does not decode or redistribute any embedded payload.

## M1.4 Flat Geometry And UV Selection

`tools/asset_compiler/terrain_geometry_resolver.py` follows the installed ArtDefs and emits `civ6_grassland_geometry_uv.json` plus the source-agnostic `samples/geometry/flat_terrain_patch.json` fixture.

The deterministic selection chain is:

1. `Terrains.artdef` entry `TERRAIN_GRASS` explicitly declares `TerrainType=Flat` and `TerrainSubType=Grass`.
2. `TerrainStyle.artdef` collection `StandardFlat`, entry `Default`, explicitly binds `GrasslandMtl` to `ART_DEF_TERRAIN_MATERIAL_GRASSLAND`.
3. The same style separately binds `GrasslandElement` to `ART_DEF_TERRAIN_ELEMENT_CONTINENTAL_HILL_GRASSLAND`. This is authored relief, not the topology of the flat base surface.
4. C3X therefore represents the flat base with a generated one-tile grid: four vertices, two counter-clockwise triangles, +Z normals, and full-range `uv0` with an explicitly declared upper-left origin.

This is a normalization decision rather than a claim that Civ VI contains an authored flat quad. It gives M1.5 a concrete, tested base topology and UV domain while keeping the runtime free of Civ VI formats.

The local inventory confirms:

- `TerrainAssetSet_Base.blp` and `TerrainElementSet_Base.blp` contain reflected vertex/index/FGX geometry types for authored assets and relief.
- `TerrainMaterialSet_Base.blp` contains the grassland material and textures, not a flat mesh definition.
- The installed Base tree contains no loose `.fgx`, `.cn6`, `.glb`, `.gltf`, `.fbx`, or `.obj` geometry files.
- CivNexus6 and Blender scripts remain source references for future authored relief import, but they are unnecessary for the procedural flat base.

Portable tests cover ArtDef selection, separation of the base material from the relief element, normalized topology, winding, normals, UV coverage, package-header inventory, and deterministic reporting. The local gate compares the installed ArtDefs and cooked-package inventory to the committed report without reading cooked payloads.

## M1.5 Normalized Pack And Textured Preview

`tools/asset_compiler/grassland_pack_builder.py` is the one-command local M1 build. It revalidates the M1.3 base-color range, reads only that embedded payload, wraps it in a standard DDS/DX10 header, combines it with the M1.4 mesh in a source-agnostic pack, and produces two deterministic PNG previews. Generated packs, DDS files, build reports, and previews remain under ignored `Renderer/packs/` and `Renderer/preview/out/` paths and must not be redistributed.

`preview/render_textured_patch.py` is source-agnostic. It consumes only `c3x.asset_pack.v0`, `c3x.normalized_mesh.v0`, `c3x.material.v0`, and standard BC3 DDS data. Its dependency-free triangle rasterizer projects the normalized mesh through the pack's isometric basis, interpolates UV0, samples BC3 blocks, and writes deterministic RGB PNGs.

Installed-asset results:

- Base color: 4096x4096, 11 mips, DXGI 78 (`BC3_UNORM_SRGB`), 22,369,616 payload bytes.
- 640x480 preview: 212,480 non-background pixels and 443 colors.
- 1024x768 preview: 262,144 non-background pixels and 443 colors.
- The pack validator rejects absolute paths and source-specific `CIVBLP`/BLP/Civ VI strings from runtime-facing JSON.

Portable synthetic tests exercise the same embedded-range-to-DDS, normalized-pack, BC3-sampling, mesh-rasterization, path-safety, PNG-writing, and two-size render path without containing Firaxis data. This completes the M1 grassland import gate; hills and the separately referenced continental-hill element remain later terrain-production work.

For actual model conversion, prefer one of:

- A Civ VI SDK/Pantry install containing loose `.fgx`, `.mtl`, `.tex`, and `.dds` assets.
- Permissioned source assets from an open Civ VI mod repository.
- User-authored Blender/GLB assets that already follow the C3X pack convention.

## Tooling References

- CivNexus6: `Renderer/third_party/CivNexus6`
- Blender import/export scripts: `Renderer/third_party/Nexus-Buddy-2-Blender-Scripts`

The intended eventual chain is:

`loose .fgx -> CivNexus6/CN6 -> Blender headless -> normalized .glb -> C3X pack manifest`
# M6.6 terrain-element height resources

The earlier geometry limitation does not apply to all terrain relief. `TerrainElementSet_Base.blp` contains a reflected `BLP::BlobEntry` array whose named `.TEH` payloads are square R8 height fields. M6.6 structurally resolves and validates `TER_Hills_Standard_Element_1` plus five standard and four desert `Mountain_*_HM_0` resources. The source compiler checks typed pointers, bounded big-data offsets and sizes, square dimensions, and the stored FNV-1a name hash before emitting generic R8 DDS files.

This proves source-derived height relief without teaching the runtime about CIVBLP or Firaxis resource names. Complete cooked vegetation-set normalization is proven separately below.

## L8 forest and jungle cooked geometry

The installed `Features.artdef` and `Clutter.artdef` files resolve `FEATURE_FOREST`, `FEATURE_FOREST_SNOW`, and `FEATURE_JUNGLE` to `environment/clutter.blp`. That package contains the expected authored component names, reflected model/container, vertex-buffer, index-buffer, material, and texture types, plus the corresponding tree and jungle texture families. These are the correct upstream assets for Terrain Lab vegetation.

A bounded scan finds 53 valid Granny/FGX headers inside the package. Directly loading the exact `Jungle_Palm_01_Anim.gr2` payload and a second generic embedded payload through the checked-in CivNexus6 dependencies reports zero models and zero meshes in both cases. The embedded FGX route is therefore animation-only for the tested ranges and cannot provide production vegetation geometry.

`tools/asset_compiler/clutter_blp_extractor.py` now locates the larger static allocation table without fixed file offsets and follows exact `LandmarkPackageEntry -> BLPPtr<IUserData> -> BaseModelData_Entry` chains. The base model binds one each of `FGXModel::ContainerDesc::{Model,Mesh,PrimGroup,Material}` for all 22 desktop vegetation bodies selected by the three ArtDef sets. Each primitive group's typed `ModelPrimGroupData` supplies its vertex-buffer index, index-buffer index, first index, index count, base vertex, and vertex count. Bounds, index coverage, non-degenerate triangles, buffer ranges, texture classes, ArtDef coverage, and runtime path independence all fail closed.

Representative verified results are:

| ArtDef landmark | Vertex format / stride | Vertices | Triangles | Material family |
| --- | --- | ---: | ---: | --- |
| `Tree_Pine_01` | `0x315CFCD9` / 24 | 47 | 39 | pine |
| `Jungle_Palm_01` | `0x6679B170` / 32 | 164 | 181 | jungle |
| `Jungle_Plant_01` | `0x6679B170` / 32 | 140 | 141 | jungle |
| `Jungle_Grass_01` | `0x315CFCD9` / 24 | 62 | 56 | jungle |

Both vertex profiles encode position as three half floats at byte 0 and UV0 as two half floats at byte 8. An earlier single-palm diagnostic incorrectly treated byte 20 as UV data; it was rejected and rebuilt before acceptance. Source tangent-frame packing remains deliberately undecoded, so the normalized meshes use deterministic area-weighted normals recomputed from the real indexed triangles. Material records bind `Generic_BaseColor`, `Generic_Gloss`, and two `LEAN` textures, which are extracted from their exact standalone `SHARED_DATA` entries and wrapped as standard DDS.

The full result contains seven base-forest bodies (three pines, two pine clumps, and two shrubs), five snow-forest bodies (three pines and two pine clumps), and ten jungle bodies (four grasses, three palms, and three plants). One shared 12-source-units-per-tile conversion preserves their relative authored sizes. Generic placement recipes retain the exact ArtDef scale, count, variation, rotation, overlap, priority, decal, width, center-model, low-end-reduction, and minimum-count values, including duplicate placements that reuse one normalized mesh.

The ignored `VegetationNormalized` pack contains only generic relative paths, C3X schemas, normalized coordinates, standard DDS textures, feature roles, and placement values. Installed paths, package offsets, source names, and hashes live only in `preview/out/clutter/vegetation_build.json`. All 33 unique upstream asset names are accounted for: 22 vegetation bodies are normalized and 11 rock, decal, or iOS-fallback entries are explicitly routed or excluded. The labeled full-size and thumbnail contact sheets consume only the normalized pack and verify every accepted body in two views with the corrected UV atlas and shared relative scale. This completes L8; L9.4 is the vegetation composition pass.

## M6.7 water, shore, river, and remaining material import

`tools/asset_compiler/water_pack_builder.py` now performs the complete local Base water import used by Terrain Lab and the native renderer. Standalone `CIVBIG` validation was extended beyond block compression to the three linear formats actually present in the water set: DXGI 10 (`R16G16B16A16_FLOAT`), 11 (`R16G16B16A16_UNORM`), and 35 (`R16G16_UNORM`). DDS headers use row pitch for linear resources and linear size for BC resources; synthetic tests cover all three new layouts.

The normalized local bundle contains 70 renderer-relevant Base textures plus 27 useful Expansion 2 textures when that source root is installed:

- large, small, secondary-small, and river lean-map pairs;
- tiling/non-tiling masks and normal channels plus water gloss;
- every observed coast, lake, island, opaque, river-source, tropical, and default dark/scatter density ramp, retaining hash-distinguished duplicates as explicit variants;
- crash foam, ripples, splashes, turbulence, mist, and waterfall support;
- beach, cliff, ocean decal, river-source decal, and river-clutter support channels;
- snow decals plus flood, flood-plain, submerged-coast, flood-wave/rapid, terraced-spring, and volcano channels;
- the three coast/ocean/river terrain-water references.

The three Expansion 2 `Terrain_EditHeightmap` blobs are a distinct bounded layout rather than standard texture `CIVBIG`: a 512-byte header/metadata region followed by a 256x256 big-endian 16-bit grid whose observed high byte is always zero. The importer validates the signature, type name, dimensions, declared payload relationship, exact trailing grid size, and 8-bit range before emitting the low byte as generic R8 DDS for flooded coast, submerged coast, and flood plains.

`TerrainElementSet_Base.blp` also yields 20 typed square R8 channels that were previously unused: two LODs each for oasis height/blend/region ID, river-bank noise, and generic/hill/mountain river-origin height and blend. The existing FNV/range/dimension checks apply before generic R8 DDS emission.

Finally, the terrain pack builder retains all eight useful material records that were not selected as a primary Civ III identity: desert hills, island base, ice, plains-hills top, river bed, salt flat, snow, and tundra base. They appear as an auxiliary generic material library rather than being forced onto semantically incorrect tiles.

Generated assets and provenance reports remain ignored and local. Runtime JSON contains only generic relative paths and roles; installed paths, package names, source hashes, and ArtDef parameter evidence remain in the ignored build report. Native water rendering now consumes the normalized large/small lean maps and foam map, replacing its previous trigonometric wave and foam stand-ins when the complete water block is present.
