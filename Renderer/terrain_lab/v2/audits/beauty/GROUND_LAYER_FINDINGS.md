# Ground layer audit: incomplete and actionable

The user requires all applicable Civ VI ground textures and correct layering
to be audited before further richness tuning. This is an explicit part of
the active surface-richness goal, not a count-based acceptance gate.

`qa/ground_layer_audit.py` inspects the installed Base/DLC and selected
environment-skin ArtDefs, normalized material and auxiliary material
descriptors, water catalog, and the actual preserved inland r3 packet.
`GROUND_LAYER_AUDIT.json` contains texture payload hashes and real draw/slot
bindings. Complete source parameters are preserved locally at the hashed
inventory path in that report. Re-run from the repository root with Python 3;
optional source-root arguments support other installations.

The first inventory contains 53 ArtDef files and 224 normalized descriptor
channels, 116 of which are absent from this one packet. These are NOT 116
proven bugs: records include aliases, unused terrain families, water effects,
and prepared future assets. Uploaded channels are also not automatically
correctly used. Source merge order, conditional selection and shader sampling
must still be traced. Fog remains Civ III-owned; natural wonders stay deferred.

## Confirmed gaps

| Source evidence | Current combined scene | Required investigation/fix |
| --- | --- | --- |
| Selected `StandardFlat` names `GrasslandHighMtl` and `PlainsHighMtl`; normalized elevated descriptors include color/height/specular | Grass-top color and height and plains-top color are not uploaded | Restore applicable high-ground layers with coherent masks and matching channels; source thresholds/geometry units need validation |
| Selected `StandardHills` names separate grass/plains high materials and height thresholds | Grass-hill-top color/height are not uploaded; current broad grass treatment does not implement this source layer graph | Trace flat/high/hill roles independently; avoid merely applying a different grass everywhere |
| Mountain stripe and snow material descriptors contain their own height/specular channels | Several are absent, even though their colors are sampled | Bind and blend complete material sets using the same weights and projection |
| Current rock color uses world triplanar projection | Existing rock height and specular paths sample different UVs | Align material coordinates before tuning bump or reflectivity |
| Grass decal source is visibly an atlas of distinct grass/rock patches | `macro_decal_uv` returns a repeated full sheet; `sample_land_clutter` samples it directly | Recover per-variant atlas coordinates and source placement; stop treating the atlas as a seamless tile |

The atlas issue also exists in normalized input: every inspected grass decal
descriptor has `uv_rect = [0,0,1,1]`. The generic decal compiler writes that
rectangle; its bounds-only normalization does not establish the individual
atlas regions. This is an importer/representation defect to investigate,
not something to hide with random UV offsets. The actual source atlas preview
is `out/ground-layer-audit/grass-decal-source.png`.

Selected overlay grass clutter has 13 named variants, including zero-count
entries and heavily weighted variants 12/13. Plains includes dark decal variants.
Existing full-sheet repetition does not reproduce that selection/placement.
Counts, scale, rotation, priority and `ShowDecal` are confirmed ArtDef values;
their exact engine evaluation and layer ordering are not yet proven.

## Rejected first diagnostic

`surface-richness-r1/inland` contains four shader-only replays on identical r3
geometry/shadow packets. A footprint-aware source-height surface gradient
barely improves flat ground and adds dark flecks on some hill crests. Reject
as a visual candidate. At noon/zoom1, 39,934 pixels change within bounds
`[58,55,1254,698]`; changed pixels alone do not make it an improvement.
The guarded shader code remains opt-in for diagnosis. All four disabled-branch
control PNGs are byte-identical to r3. No previous best was overwritten.

The [decal reconstruction pass](GROUND_DECAL_PASS.md) has now recovered exact
triangle/UV data for 19 grass/plains variants and traced the selected override's
four associated textures. Twenty matched diagnostic frames include a wholly
unseen 100-tile region. The gain is subtle and the fragment experiment is too
expensive; no new best is promoted. Placement/compositing semantics and the
remaining high-ground/mountain layer graph are still unproven.

Next: resolve source layer bindings and a practical patch representation; then
implement and compose bounded corrections, inspect the fixed benchmarks and
a newly selected holdout, and reassess the three largest remaining visual gaps.

The [mountain channel pass](ROCK_CHANNEL_PASS.md) now aligns projected height
and specular with color and adds eight missing snow/stripe channels in copied
combined packets. Sixteen matched frames show modestly clearer gray rock
relief. All 72 test.biq mountains have grass base terrain, so desert stripe
coverage uses an explicitly synthetic material witness. Production bindings,
source blend semantics, crop/wrap stability and cost remain open. Source
`StandardFlat` also references continental terrain elements; trace their
geometry with the high material layers before further flat-ground tuning.

[The continental/source-baking pass](CONTINENTAL_GROUND_PASS.md) recovers those
height fields and rejects a pale high-layer diagnostic. Installed shader
bytecode now proves that source BaseColor alpha participates in shared
alpha-squared material weighting before normalized cache resolution. The final
terrain stage consumes cached normal and specular/AO data, with additional AO
and shadow inputs. Merely binding all source DDS files does not reproduce this
processing. The first alpha-weighted scene diagnostic changes a limited terrain
transition; cached height-to-normal and AO generation remain the next audit.
