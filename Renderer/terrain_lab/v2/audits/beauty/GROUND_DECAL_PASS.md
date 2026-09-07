# Ground decal reconstruction: diagnostic, not promoted

The source audit now recovers the actual triangle geometry and atlas UVs for
19 nonzero-count grass/plains variants (297 packed vertices). Base uses
108-byte `DecalDesc2` records; the selected environment override uses 92-byte
`DecalDesc` records. Reflection names, typed vertex-buffer references, index
ranges and bounds are checked by `systems/terrain/prepare_ground_decals.py`.
The two packages have identical triangle/UV data for all 19 variants. Decoded
Base geometry agrees with its exact content bounds within 0.00657 source units.
This resolves the former full-sheet UV assumption for these variants.

The selected override's grass/plains color and height DDS files are extracted
and hashed alongside the geometry. Local derived art/data remain ignored;
`fixtures/beauty/source-ground-decals-r1/provenance.json` records reproducible
identities. Replacing only these four textures changes just 366 inland noon
zoom-1 pixels by at most one channel value relative to the Base-texture
diagnostic. Their alternate payloads do not explain the missing richness.

Confirmed source data and Lab interpretation remain separate: source triangle
UVs, variant counts, scale and variation are recovered; the conversion to Civ
III units, density, random placement algorithm, priority evaluation and final
engine compositing are not established. The generic shader evaluates explicit
triangles and uses analytic UV gradients. Its world-space placement is an
experimental adapter, not a recovered Civ VI algorithm. It does not complete
the user's all-applicable-ground-layer requirement.

## Render loop and decision

- r1: source triangles and UVs, inherited opacity, Base textures. Too subtle.
- r2: remove the extra inherited opacity attenuation. More patch variation,
  still weak ground relief.
- r3: use the selected override's four DDS files. Almost visually identical.
- r4: larger patches and lower cell density. Twenty matched noon/midnight,
  zoom-1/zoom-2 frames across coastal, inland, wilderness, freshcanopy and the
  newly selected freshground region. No promotion.

At actual gameplay size, r4 changes the grass/soil distribution in the bare
central field of freshground and beside its lower river. Some small patches
are more recognizable, but the field still reads as a mostly flat colored
surface. It does not reproduce the closely spaced grass relief, exposed soil,
and rough rocky transitions visible in canonical `sea_and_shore.png`.
Canonical `mountain.png` also has coherent fine cracks over broad faces;
our surviving directional bands and uneven face detail remain unresolved.
Reference closeups have more pixels per feature and are diagnostic references,
not a claim of matched camera scale.

[Freshground native comparison](out/surface-decals-r4/review/freshground-noon-native.png)
shows baseline above and r4 below, both unscaled 640x320 crops. Changed pixels
are evidence of the effect, not proof that the effect improves the scene.

| Noon / zoom 1 | Changed pixels | Bounds | Baseline / diagnostic GPU ms |
| --- | ---: | --- | ---: |
| Coastal | 118392 | 269,59–1094,699 | 39.28 / 213.25 |
| Inland | 156355 | 58,55–1320,694 | 46.30 / 211.69 |
| Wilderness | 120923 | 40,63–1088,699 | 30.34 / 181.51 |
| Fresh canopy | 94611 | 101,62–1129,656 | 33.36 / 212.39 |
| Fresh ground | 127378 | 268,76–1320,578 | 62.85 / 194.16 |

These local timing samples are not a controlled production benchmark, but
the fragment loop's repeated variant selection, triangle search and height
sampling are clearly unsuitable for promotion. A future implementation should
compose the patches into a cached generic material field or rasterize their
geometry once, then sample coherent color/height from that representation.
Do not continue tuning fragment-loop parameters as the main richness strategy.

## Holdout and engineering evidence

`beauty-freshground-100-v1`, origin [68,26], was selected from test.biq before
viewing. Every one of its 100 tiles lies outside previous beauty crops. It has
25 grassland, 15 plains, 3 hills, 7 mountains, 21 forest, 19 coast and 10 sea
tiles, with 17 river tiles. Its camera, halo and source identities are frozen in
`fixtures/beauty/surface-decals-foundation/freshground/BENCHMARKS.json`.
No local adjustment followed inspection; it is now a regression witness.

The initial foundation render used the intermediate export recipe after a
preparation failure. Exclude `out/surface-decals-foundation/freshground` from
all comparison and acceptance. The corrected merged recipe was rendered to
`out/surface-decals-foundation-v2/freshground`; r4 uses that baseline. Both
histories are retained and no previous best was overwritten.

`qa/verify_ground_decals.py` verifies four byte-identical disabled-branch
control BMPs, the unchanged original packet hashes and copied packet hashes,
four intended texture replacements in each of 20 packets, matched output
sizes and changed-pixel bounds. Rebinding preserves the complete packet tail
containing geometry, draw bindings and shadow structure byte-for-byte.
[Machine evidence](GROUND_DECAL_r4_EVIDENCE.json) records these checks.
`python3 Renderer/tools/renderer_dev.py lab` passes 132 Python and 12 Node tests.
These checks do not grant visual acceptance or close LQ0.

## Next three visible gaps

1. Missing flat/high/hill material roles and masks leave grass/soil transitions
   too uniform. Trace the effective source graph and bind complete channels.
2. Mountain color, height and specular need the same projection and blend
   weights before more bump-strength tuning.
3. Small surface relief remains weak after filtering to gameplay pixels.
   Reassess physical texture scale and filtered height only after the layer
   graph is coherent; a cached patch representation may supply useful detail.

Preserve river-corridor-r3 and the broader shadow-receiver-r1 checkpoint. The
frozen Integration preparation remains unchanged; this experiment is not a
replacement pickup package. All milestone and human-review gates remain open.
