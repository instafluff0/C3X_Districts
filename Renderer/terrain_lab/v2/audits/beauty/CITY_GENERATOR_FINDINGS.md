# City generator and grounding evidence

The user identified organized city fabric and ground textures in a canonical
Civ VI city image and explicitly deferred roads. Installed city metadata confirms
both. The former importer read only `GeneratorBlockList`: its “complete graph”
description overstated coverage of city organization. It did not consume the
`Generator` or `GroundingMaterials` collections.

## Authored organization

`tools/asset_compiler/city_generator_probe.py` now preserves those collections
from Base and Expansion2, the two installed CityGenerators documents with
nonempty generator/grounding definitions. Other installed CityGenerators files
add block/culture choices but have no nonempty generator records in this probe.
The source report and generic parameter profiles are in
`fixtures/beauty/city-generator-source-r2/` (ground triangles remain in r1).

The city-center generator defines a hex/spine mode, spine width and length
ranges, model scale, block variation and height range, population-dependent area,
filler occupancy, filler ratio and scatter grouping. Its art-era distributions
include explicit `OrderFromCenter` and `Weight` values. For Base modern cities:

| Contributing art era | Order from center | Weight |
| --- | ---: | ---: |
| Modern | 0 | 0.5 |
| Industrial | 1 | 1.0 |
| Classical | 2 | 1.0 |

Those are authored weights, not proven exact building-count percentages. The
existing C3X fallback calls source Classical “medieval.” The prior all-modern
selection omitted the older outer fabric entirely. Population records at 0, 1,
14 and 22 specify area and fill changes; their source-engine interpolation and
physical units are not established. Expansion2 changes spine length ranges
from Base's 1.1–1.6 to 1.3–1.8 and adds a future-era distribution. Its modern selector is `ARTERA_MOD_NO_FUTURE`,
while the modern layer weights/order match Base; r2 preserves that mapping and
collection merge controls. r1 omitted the selector alias and is superseded for
normalized generator parameters. The profiles are
kept separate rather than silently guessing an active content merge.

The exact placement algorithm is not contained in this XML. An adapter still
must preserve Civ III diamonds and authoritative anchors, fit complete blocks,
keep growth stable, and respect terrain/clearance. Neither copying a hex camera
nor inventing a universal radial scatter reconstructs that algorithm.

## Ground textures and recovered triangle selection

`GroundingMaterials` selects era-specific normal and pillaged materials from
the source route-decal material library, including `Decal_Parts_Modern_01`.
This establishes shared material use; it does not prove route path generation
or authorize road implementation. No connecting-road code was changed.

The existing normalized city compounds also include ground descriptors, omitted
by the city scene producer. Their base-color atlases contain asphalt pads,
paving/parking markings and dirt surrounds. These are sheets of parts, not
textures to stretch wholesale over a footprint. The bounds-only importer lost
the atlas selection geometry.

`qa/prepare_city_ground_probe.py` reuses the proven ground-decal decoder with
explicit descriptor indices. The selected modern rectangular block has two
six-vertex descriptors: one points to the modern paving atlas and the other to
the ancient dirt atlas. The exact source XY/UV triples and hashes are retained.
The relationship between descriptor index and operational/state selection is
still unproven; r16 explicitly probes descriptor 0 and does not claim automatic
normal/pillaged selection. Height response also remains unimplemented.

The r16 scene includes only that compound's source paving, unchanged buildings,
and the fixed 100-tile terrain. Source triangles are subdivided with barycentric
UV preservation and projected onto authoritative sampled terrain; they receive
shadows but do not cast shadows or write depth. Treating the paving as a rigid
building foundation initially failed the height gate; projection resolved the
actual issue without weakening the building gate. All sampled paving vertices
remain dry. Noon/midnight at both zooms render using shared-resource packets.

At normal gameplay size the visible gain is only a narrow paved edge: 59 noon
and 53 midnight pixels differ by more than two channel levels from the matched
addressing-only control. Most of the patch is occluded by the packed towers.
This is recovered input and a bounded composition proof, **not a new accepted
city best**. Broad coherent urban ground needs the generator's layout and
material-layer logic together. Source atlas and matched views:

- `out/city-material-r1/source-ground-decals.png`
- `out/city-scene-r16/review/ground-native.png`

## Experiments superseded by this finding

r14 corrects clamped sampling of source repeat UVs and tests diffuse slope-map
detail. The repeat fix changes 82 noon city pixels; adding slope detail changes
303 more. Medieval and modern closeups show faint facade/roof detail, but the
gameplay-size improvement is too small to claim restored materials. Both remain
opt-in. Gloss, auxiliary AO coordinates and the authored tangent frame remain
open. The [LEAN paper](https://userpages.cs.umbc.edu/olano/papers/lean/) explains
filterable normal statistics, but does not prove Civ VI's packing or shader.
The current diffuse experiment is an adaptation, not its recovered BRDF.

r15's lower-to-taller ordering produces town and city views, but its metropolis
cannot fit the late complete neighborhood block. It is rejected as a complete
growth recipe. Stop tuning that ordering in isolation: use the recovered era
layers, block roles and city area/fill metadata for the next combined layout.
The r8 night, r11 growth and r13 palace checkpoints remain preserved.

The historical next step was metadata-led mixed-era composition. The user
selection below supersedes it: use one era with correctly selected ground
pieces, then size/culture/terrain and night/reflection checks. Roads remain deferred. No native city ownership, milestone advancement
or human approval is claimed.

Verification: 16 focused importer tests pass, ground-part regeneration matches
the saved input exactly, and four standalone Metal/D3D11 ground comparisons
pass. `CITY_GROUND_r16_EVIDENCE.json` records matched unchanged buildings, pixel
differences and parity. No full Lab pass is claimed; the previous L19A input hash
gate failure remains unresolved by this work.

## User selection: one era per city

The user reviewed r17 and explicitly prefers the pure single-era look, closer
to Civ III. The multi-era composition is rejected regardless of source fidelity.
`--generator-profile` now defaults to the current era only and preserves the
previous component sequence and placement scoring. `--historical-era-mix` exists
only to reproduce the rejected diagnostic; it is not a selected workflow.

Growth, density, block placement and grounding metadata remain useful within
one era. The next pass must keep that visual policy. The r17 mixed city actually
occupies 0.510 tile-squared of bounding-box footprint versus 0.471 previously;
the initial impression of reduced density came from spacing/occlusion, not less
occupied area. Do not use the earlier density explanation as evidence.

## Subsequent single-era ground and capital pass

See [r18/r19 findings](CITY_GROUND_AND_CAPITAL_PASS.md). Exact ground recovery now
covers all selected components in two pools, and shoreline clipping handles
source pads outside legal building footprints. The new 47-root palace pack is
connected through an explicit American modern mapping. The initial capital
layout required bounded alternative palace sites; its remaining facade occlusion
and scattered skyline prevent a new city-best claim.
