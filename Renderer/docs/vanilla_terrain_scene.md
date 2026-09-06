# M6.6 Vanilla Terrain Scene

> Historical note: this document records the M6.6 intermediate scene. The
> approved L9-L11/I9-I11 path now suppresses `m19` wholesale when custom
> rendering is on, reports zero production fallback, and never restores native
> terrain after failure.

M6.6 replaces the flat-material bootstrap with a source-independent, depth-tested terrain surface while keeping incomplete families atomic.

## Runtime composition

- `terrain_type` selects the underlying ground material.
- `real_terrain_type` selects flood plain, hill, mountain, volcano, and water surface composition.
- Forest, jungle, and marsh retain their complete Civ III feature bodies above the custom underlying ground.
- After a successful custom blit, the existing `m19` hook suppresses Civ III's `0x4010` mountain/hill/volcano dispatch only for the exact captured screen instances accepted by the renderer. Per-tile and whole-frame fallback retain native relief.
- Polar ice, authored shoreline detail, and landmarks remain Civ III-owned until their complete geometry and selection dependencies are normalized.
- Roads, rivers, resources, cities, units, borders, fog, labels, selection, HUD, and UI remain Civ III-owned.

This fixes the earlier semantic error where a grassland mountain was interpreted only as flat grass. Feature flags are now derived from `real_terrain_type`.

## Local source compilation

`terrain_relief_builder.py` follows reflected `TerrainElementSet_Base.blp` allocation metadata and validates typed `BLP::BlobEntry` names, byte ranges, perfect-square R8 payloads, and FNV-1a hashes. It compiles:

- `TER_Hills_Standard_Element_1` to generic surface-detail R8 DDS.
- Five standard and four desert `Mountain_*_HM_0` resources to a generic 3x3 R8 DDS atlas.

The terrain pack now contains all fourteen generic logical terrain IDs. It uses ordinary BC3 base-color DDS and R8 height DDS resources; no runtime path, format, or code references Civ VI. Converted source payloads stay local and ignored.

## Rendering

The native pass tessellates each tile with a screen-density-aware grid (16x16 at 128px tiles and 8x8 at 64px), samples normalized height fields on the CPU, projects through Civ III's authoritative pixel anchors, and writes a D24S8 depth target. UVs are defined in map space so adjacent tiles share exactly the same texture coordinates along their common edge. Shared world-space evaluation keeps geometry deterministic across consuming tiles and horizontal wrapping.

Water uses source water material texture plus bounded generic depth tint and wave modulation. Material grading is applied to the source textures; it does not replace them with procedural colors.

## BIQ validation

`export_biq_terrain_scene.js` delegates BIQ parsing to the neighboring C3X Editor and emits a minimal source-independent terrain CSV. `biq_preview.exe` sends those records through `C3XRenderer.dll`, using the same frame ABI, definition resolver, terrain selection, D3D target, projection, and readback as the injected game path.

The user-supplied `Conquests/Scenarios/test.biq` is a 100x100 map with 5,000 tiles. M6.6 deterministically renders:

- `Renderer/preview/out/m6_6_test_biq_128.bmp` at 1600x900 and 128x64 tiles.
- `Renderer/preview/out/m6_6_test_biq_64.bmp` at 1280x720 and 64x32 tiles.
- `Renderer/preview/out/m6_6_all_terrain.bmp`, which exercises all fourteen terrain identities.

The generated screenshots are local/untracked. Verification requires zero fallback for these scenes, material diversity, nonblank coverage, byte-identical repeat output, and a successful injected-code compile.
