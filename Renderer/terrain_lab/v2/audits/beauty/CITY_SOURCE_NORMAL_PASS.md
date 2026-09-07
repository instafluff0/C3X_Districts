# Packed city normals: isolated r29/r30 diagnostic

The source-normal experiment changes roof edges and wall corners slightly, but
does not establish a new visual best. Keep r28 as the preceding material
candidate. City quality, source material completeness and all milestone gates
remain open. This is standalone Lab work; no runtime pack or injected code changed.

The earlier city importer recomputes area-weighted geometric normals. For the
tested static vertex profile `0x315CFCD9`, stride 24, bytes 6–7 (after half3
position) closely fit a signed-byte octahedral normal encoding. The twelve
European medieval source components produce 22 distinct normalized meshes.
Every tested direction has positive dot product with the recomputed normal;
primitive mean agreement exceeds 0.97. Some individual corners differ much
more, consistent with authored smoothing but insufficient to prove the original
engine's entire shading convention. The exporter rejects other profiles.

The neighboring packed pairs are not yet a verified tangent basis: some are
degenerate and fail orthogonality. No tangent, bitangent, LEAN texture or gloss
interpretation is promoted by this experiment. A general explanation of
octahedral direction encoding is available in the
[unit-vector representation survey](https://jcgt.org/published/0003/02/01/paper-lowres.pdf);
the installed-byte evidence, not that paper, supports this particular offset.

`qa/prepare_city_source_normals.py` emits a generic per-mesh normal mapping.
Geometry fingerprints exclude normals and include positions, topology and all
three coordinate sets. `qa/city_scene_pass.py --source-normals` validates those
fingerprints and finite unit-length arrays before replacing only normals.
Unmatched meshes are recorded explicitly. This remains opt-in and does not
rewrite the shared normalized asset packs. Source data is regenerable and ignored.

## Matched visual evidence

Coastal r29 uses exactly r28's single-era layout, source scale, paving, UV1 AO,
UV0 diffuse repeat addressing, UV2 emission, lighting and cameras. At normal
gameplay size 835 noon pixels and 328 midnight pixels differ by more than two
channel levels. The change is confined to the city. The comparison is
`out/city-material-r2/source-normals-native.png`; roofs and wall corners account
for the difference. Increased numerical agreement alone is not visual acceptance.

Inland r30 repeats the unchanged r27 city/material witness at tile (7,4), with
the same normal mapping and no regional tuning. The frozen 100-tile terrain and
object placements remain intact. Both regions retain their existing reference
and candidate images. Fixed wilderness city coverage is still unfinished.

`qa/city_source_normal_evidence.py` verifies all non-normal vertex bytes and
texture bindings, placement/material parameters, unchanged shader/lighting
closures, local pixel differences and six successful Windows D3D comparisons.
Ten focused decoder, ground clipping and palace importer tests pass. The decoder
tests cover known axes, lower-hemisphere folding and all 65,536 byte pairs.
[Recheckable evidence](CITY_SOURCE_NORMAL_r29_r30_EVIDENCE.json).

Compared with `Renderer/canonical/nightlights.jpg`, the three larger remaining
gaps are material separation across roofs/walls, coherent urban ground coverage,
and light reaching nearby surfaces. Mesh normals alone do not supply those.
Continue with the actual tangent/normal-texture/gloss bindings and grounding
layers rather than increasing superficial shading strength. Preserve the
single-era preference, explicit palace mapping and deferred connecting roads.

To reproduce the source mapping, use a new output path:

```bash
python3 Renderer/terrain_lab/v2/qa/prepare_city_source_normals.py \
  --pool european/medieval --output NEW_NORMAL_MAPPING.json
```

Add `--source-normals NEW_NORMAL_MAPPING.json` to the r28 command documented in
`CITY_AO_MATERIAL_PASS.md`, with a fresh revision. For the inland comparison,
use `--region inland --anchor 7 4`, omit `--source-addressing` to match r27,
and omit `--all-zooms` for its two normal-size frames.
