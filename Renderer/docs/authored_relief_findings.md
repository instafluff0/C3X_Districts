# Authored Relief Findings

## Scope

M6.7c1 examines the installed terrain-element package used by the local development importer. The repeatable probe is:

```powershell
py Renderer/tools/asset_compiler/terrain_relief_builder.py `
  --output Renderer/preview/out/m6_7c1_relief_analysis `
  --report Renderer/preview/out/m6_7c1_relief_analysis/report.json
```

The generated report and contact sheets are local evidence and are not distributable pack content.

## Confirmed Package Data

- Five standard and four desert mountain variants each contain aligned `HM`, `HBLEND`, and `ID` resources at 256x256 (`_0`) and 128x128 (`_1`).
- `HM` is the authored macro-height channel. Its contact sheet exposes asymmetric multi-peak forms that the runtime must preserve.
- `HBLEND` is continuous 8-bit data with a bounded footprint. Standard variant 1 has 256 unique values and only 18.4% nonzero edge samples; its `_0`/`_1` box-downsample correlation is 0.999691.
- `ID` is discrete 8-bit data. Standard variant 1 uses four values and desert variant 1 uses six; both are zero at their image boundary.
- Hill element pairs are 512/256 for `standard` and 2048/1024 for `continental`, `continental_plains`, and `continental_snow`.
- Tested `_0`/`_1` box-downsample correlations range from 0.993042 through 0.999691. This confirms aligned content and strongly supports a high/low LOD relationship rather than independent variants.
- Installed `TerrainStyle.artdef` defines standard mountain height 32, width 42.399979, snow-low 24, and snow-high 26. The separate desert mountain definition uses height 24 and width 54.

## Inferred Semantics

- `_1` is a lower-resolution LOD of `_0`. The dimensions and measured correlations make this a high-confidence inference, but no inspected field explicitly names the pair as LODs.
- `HBLEND` is a footprint or height-blending field. Its exact equation and whether it modifies footprint, height, or both remain unconfirmed.
- `ID` identifies material or geometric regions. Exact value-to-material bindings remain unconfirmed.
- Standard and desert mountain families are authored selection groups and must remain separate in normalized data even if a future pack deliberately reuses shapes across biomes.

## Unresolved

- The exact `HBLEND` composition equation.
- The meaning of each nonzero `ID` value and its relationship to upper rock, snow, and desert stripe materials.
- The intended selection relationship among standard and continental hill fields.
- Runtime placement rules used by Civ VI to combine individual mountain contributions into ridgelines.

## Implementation Consequences

Terrain-element height defines macro geometry; material height defines small-scale shading detail. The normalized pack must preserve that separation. Procedural runtime work may place, orient, blend, join, and subtly vary authored contributions, but it must not replace the authored silhouette with radial cones, sine mounds, or dominant synthetic lobes.

The old combined nine-cell mountain atlas is compatibility-only. M6.7c2 replaces it with generic family/variant/channel/LOD records, after which M6.7c3 and M6.7c4 can make the authored fields authoritative in the shared surface.

## Implemented Runtime Composition

- Hills now sample the 512x512 authored field as the macro surface. The surrounding topology weights blend flat and hill control points, while high-elevation color is selected from the underlying Civ III biome rather than a universal green hill material.
- Mountains now preserve the standard five-piece and desert four-piece groups. Underlying desert/flood-plain ground selects the desert group; deterministic orientation, footprint scaling, and connection-directed overlap compose neighboring pieces into ranges.
- The former radial skirt, procedural secondary/tertiary peaks, radial cone mask, and sine edge noise no longer participate in mountain geometry.
- Standard mountains use authored base, upper-rock, and snow materials. Snow follows the confirmed 24/26 thresholds normalized by the declared height of 32. Desert mountains use their authored base and three stripe materials at the declared 10, 18, and 23 height bands normalized by the declared height of 24.
- Tiles in a relief neighborhood use a 24x24 grid at the 128px tile view and 12x12 at the 64px view; ordinary flat regions retain the 16x16 and 8x8 grids.

`HBLEND` and `ID` remain preserved in `c3x.relief_set.v0` but do not yet drive runtime equations because their exact engine semantics remain unresolved. The implemented composition therefore uses `HM` as geometry authority and documented ArtDef parameters for material placement without claiming an exact Civ VI reconstruction.
