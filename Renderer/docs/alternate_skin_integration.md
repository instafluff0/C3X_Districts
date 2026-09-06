# Alternate Terrain Skin Integration

## Current result

The renderer now has the first executable slice of the intended folder-to-pack workflow:

- `tools/import_source.py` accepts a Civ VI `Assets` root, an optional visual-skin overlay root, a variant name, and an output directory.
- The source-specific adapter selects the overlay's cooked terrain-material package and delegates to the existing normalized terrain compiler. The first slice explicitly inherits baseline relief and water.
- `tools/asset_compiler/pack_equivalence.py` compares logical assets by normalized records and recursively referenced runtime-file hashes. Its deterministic report classifies shared, replaced, inherited/unchanged, missing, and added IDs.
- An overlay import is staged before publication, refuses to overwrite an existing variant unless `--replace` is explicit, and always requires a baseline pack.
- Runtime records remain source-independent. Source names, installed paths, permissions, and adapter details live only under the pack's `provenance/` directory, which the runtime never reads.

This is a terrain-material vertical slice, not the final all-category importer. The candidate skin's `TerrainElementSet` is a partial overlay and lacks `TER_Hills_Standard_Element_0`, so it cannot replace the complete baseline relief package atomically yet. Relief and the dedicated animated-water resources currently come from the baseline Civ VI source root; coast, sea, and ocean terrain materials come from the overlay. Per-resource relief overlay resolution, alternate water packages, clutter, vegetation, decals, and the noon color key remain future adapter work and must retain the same generic runtime contracts already used by Renderer Lab and production.

## Installed Civ V Environment Skin probe

A metadata-only inspection on 2026-09-05 confirmed the locally installed Steam Workshop item `1702339134` at:

```text
$HOME/Library/Application Support/Steam/steamapps/workshop/content/289070/1702339134
```

The local `Civ_V_Art_Skin.modinfo` identifies **Environment Skin: Sid Meier's Civilization**, credits Brian Busatti, and registers replacement Terrain, TerrainStyle, Water, WaterMaterials, Wave, Clutter, Overlay, and GameLighting ArtDefs. Both macOS and Windows cooked roots contain terrain material/element, water, wave, clutter, overlay, and lighting packages.

The Windows terrain material package passes the existing typed, metadata-only CIVBLP resolver:

- 31 material records and 87 typed texture records.
- All 30 distinct material names currently requested by `terrain_pack_builder.py` are present.
- `ART_DEF_TERRAIN_MATERIAL_DESERT` occurs twice. Occurrence 0 binds `CIVV_TER_Desert_B` plus its matching height/FOW channels and is the correct base desert record. Occurrence 1 binds a mountain-desert stripe and must not win base-desert selection.
- The overlay adapter therefore uses explicit occurrence policy `{desert: 0, mountains: 0}`. Vanilla keeps its existing `{mountains: 0, ocean: 1}` policy.

No texture, height field, model, or other payload from this skin was extracted during the initial metadata probe. The subsequent user-authorized private build described below did extract the terrain-material payload into an ignored local pack.

## Local build and render evidence

The private local test build completed on 2026-09-05 at `Renderer/packs/Civ5EnvironmentSkin`:

- 827 MB of generated payload, excluded by the existing pack ignore rule.
- 14 shared logical terrain IDs, 14 replaced, zero inherited at whole-asset fingerprint level, zero missing, and zero added.
- Equivalence report hash `13df2241dc75eae49072c1d71b826e6ff029b227b0ffa2cf0e3656ecbdd84dc4`.
- Component provenance records baseline relief and animated-water resources as inherited even though each complete terrain asset fingerprint changes with its overlay material.
- The Windows Renderer Lab built and rendered all four L11 outputs through the alternate pack. The full beauty render hash is `a21f16002c33a093afe08a9627b7789ed7d4f06d29a590708241a48b31754c1e`.
- The native integration workflow also passed its layered-definition, D3D11 renderer, fallback, scheduling, ABI, and bounded-blit smoke tests. No injected C source or Civ III patch address changed.

These are implementation and local-render checks, not formal M6.8 completion or the alternate skin's visual-promotion approval.

## Local testing and distribution

The current Nexus permissions prohibit conversion to another game and require permission before asset reuse. No output from this skin may be committed or distributed by C3X without a rights-holder grant.

For the user's explicitly requested private experiment, `--local-testing-only` records a provenance notice that the generated pack is an ignored, non-distributed test artifact. A future distributable build instead requires `--permission-record` with documented `conversion` and `cross-game-use` permission:

```json
{
  "schema": "c3x.source_conversion_permission.v0",
  "source_name": "Environment Skin: Sid Meier's Civilization V",
  "rights_holder": "Rights-holder name",
  "grant_reference": "Durable URL, email archive, or other recorded grant",
  "permissions": ["conversion", "cross-game-use"],
  "redistribution": "allowed"
}
```

## One-command builds

Baseline terrain pack:

```text
python3 Renderer/tools/import_source.py \
  --source-root "/path/to/Civ6.app/Contents/Assets" \
  --variant vanilla \
  --output Renderer/packs/Civ6Vanilla
```

Local alternate terrain test pack:

```text
python3 Renderer/tools/import_source.py \
  --source-root "/path/to/Civ6.app/Contents/Assets" \
  --overlay-root "/path/to/environment-skin" \
  --variant alternate-environment \
  --baseline-pack Renderer/packs/TerrainNormalized \
  --local-testing-only \
  --output Renderer/packs/Civ5EnvironmentSkin
```

The second command writes `provenance/equivalence_report.json` inside the ignored local pack. Identical repeated inputs produce the same logical-asset classifications and fingerprints.

## Selecting the pack

Renderer Lab now preserves an externally supplied pack root:

```bat
set "C3X_LAB_PACK=..\packs\Civ5EnvironmentSkin"
call Renderer\terrain_lab\RUN_L11.bat
```

The production runtime supports the same selection through normal definition layering. Copy the single `#Pack` section from `samples/config/alternate_pack.custom_rendering.txt` to the ignored `Renderer/custom.custom_rendering.txt`, changing only its path. The existing `terrain_normalized` asset aliases and rules remain unchanged.

Neither path introduces an alternate-skin switch or a Civ VI-specific branch into the runtime renderer.
