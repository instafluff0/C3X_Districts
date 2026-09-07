# Single-era capital composition, r21/r22

Status: provisional improvement to the American modern capital relative to r19.
No Civ VI equivalence, human approval, native promotion, or milestone closure.
The complete city goal remains active. Previous overall bests and all historical
captures are preserved. r20 is superseded by r21's corrected projection scoring.

## Visible change and composition

The r19 palace was physically legal but its facade was obscured by foreground
towers. `--capital-composition` combines the existing compact house-placement
recipe with a preference against foreground screen-box overlap of the palace.
This is a generic layout heuristic, not a recovered Civ VI placement algorithm.
It ranks legal sites without changing source bodies, UVs, scales, rotations,
counts, building-water checks or the selected single-era house pool.

The first r20 probe exposed a projection mismatch: layout screen boxes used the
lowest source vertex as ground, whereas the rendered city used authored z=0.
Many sources have buried foundation skirts. r21 scores the same authored ground
plane as the rendered bodies. The comparison shows a clearer central civic
facade and groups the foreground buildings together. This is visual grouping,
not a claim that the physical bounding rectangle or occupied area shrank.

Matched no-palace controls preserve every surrounding house. Within the fixed
palace image area, the changed-pixel contribution rises from 878 to 1,082 at noon
and from 879 to 1,056 at midnight. These count pixels visibly contributed by the
palace relative to its control; they are not geometric visibility percentages.
No new palace-light contribution appears in the fixed interior lake sample.
The changed arrangement visibly changes the ordinary towers' reflections, while
the existing r8 controlled reflection evidence remains the proof of that effect.

[Layout comparison at gameplay size](out/city-scene-r21/review/capital-composition-native.png)

r22 composes this layout with the prior source-ground recovery under all seven
ordinary city components. Paving adds 724 noon and 652 midnight gameplay-size
pixels around their bases, without moving buildings. It is a modest grounding
gain, not broad urban ground reconstruction. The palace's own ground descriptor
has not been recovered in this pass.

[Combined layout and paving](out/city-scene-r22/review/combined-capital-native.png)

## Ground blocker resolved through composition

The first r22 attempt stopped before rendering: a few pad vertices were in
water-classified tiles despite a negative smoothed shore distance. These are two
distinct restrictions, not a reason to discard the building water gate.
Ground triangles now clip to exact dry tile-cell boundaries, then to the locally
interpolated smoothed shore. Every attribute, including atlas UVs, interpolates
at the cut. Unknown intersected tile cells fail explicitly. Building foundation
and water checks are unchanged. The analytic test verifies retained area and UV
parameterization when a water cell has a negative smoothed shore value.

## Growth and region check

All three American modern sizes render at the same source scale, camera and
0.95-tile half-extent. The 4/7/11 ordinary-component sequence grows by appending;
every earlier building and the palace retain exactly the same position. The
smallest stage still contains the source tower compound, so fuller visual growth
calibration remains open even though positional stability passes.

[Growth at gameplay size](out/city-scene-r21/review/capital-growth-native.png)

The producer now supports the existing `coastal`, `inland`, `wilderness` and
`freshcanopy` 100-tile fixtures instead of hardcoding the coast. Foundation query
caches and shader imports follow the selected region. This pass renders coastal
and freshcanopy only; it does not claim full fixed-benchmark city coverage.

Freshcanopy was previously used for natural-scene work but had not been tuned for
cities. The unchanged recipe first rejected anchor (3,2): sampled terrain height
varied from 2.5 to about 130. A metadata-only survey selected the nearest dry,
nearly flat 3x3 envelope to map center, breaking ties by coordinates, before
viewing the city. Anchor (5,4) then rendered with unchanged assets, scales and
recipe. Its palace is clearly visible in front of the rear forest. Full geometry
clearance against vegetation/routes still needs coverage; this image alone is
not proof of universal clearance. The fixture is now a city regression witness,
not an untuned region for later passes.

[Region new to city tuning](out/city-scene-r21/review/freshcanopy-native.png)

## Evidence, next work and reproduction

Ten standalone Windows comparisons pass: four coastal r21, two freshcanopy r21,
and four combined r22. Seven focused geometry/palace-importer tests pass.
`qa/city_composition_evidence.py` rechecks source identity, matched controls,
growth prefixes, localized pixel changes, the holdout and all parity results.
[Retained evidence](CITY_COMPOSITION_r21_r22_EVIDENCE.json).
This supporting engineering evidence does not establish visual acceptance.

Reproduce the composed case from the repository root:

```bash
python3 Renderer/terrain_lab/v2/qa/city_scene_pass.py \
  --revision NEW_REVISION --pool american/modern --size 1 --factor 1.5 \
  --expanded --authored-ground --emissive-gain 8 --emissive-uv 2 --glow \
  --source-addressing --capital --capital-composition --anchor 7 5 \
  --all-zooms --footprint-limit .95 \
  --compound-ground Renderer/terrain_lab/v2/fixtures/beauty/city-generator-source-r2/modern-ground-parts.json
```

Use a new integer revision; preserved renders must not be overwritten. The ground
map regenerates with `qa/prepare_city_ground_probe.py --pool american/modern` and
the same output path only if that ignored derivative is absent.

The largest remaining visual gaps are flat facade/material appearance, broad
ground coverage between source pads, and loss of civic facade visibility as the
metropolis fills out. Carry this recipe into other era/culture capital mappings,
compare against their previous best, and reject regressions. Single-era selection
remains mandatory. Connecting roads remain deferred. Local night light transport,
the full source material audit, authoritative capture and all existing gates
remain unfinished.

After verification, regenerable r20-r22 linear GPU intermediates were discarded
(240.7 MiB reclaimed). Reference images, controls, masks, shader snapshots,
replay packets and shared blobs remain. The evidence recheck passes after cleanup.
