# River and vegetation campaign: first combined loop

2026-09-06. Active goal, not complete or accepted. Local Metal only.
`shadow-receiver-r1` and its frozen integration preparation archive remain
unchanged. `canopy-variation-r1` and `river-corridor-r2` are opt-in candidates.
No milestone closure, human approval or Integration promotion is recorded.

## Visible results

- [Forest arrangement, coastal gameplay size](out/canopy-variation-r1/review/coastal-h12-z1.png):
  the fixed early tree pieces and missing-anchor slots no longer occupy the
  same row of each stand. Source sizes, rotations and instance counts remain
  identical; changed crown overlap and silhouettes are the visible result.
- [River and headwater pool beside inland mountains](out/river-corridor-r2/review/inland-h12-z1.png):
  the pointed stream end below the isolated mountain becomes a connected small
  pool; tight joins share smooth tangents, and the channel through the forest
  is more readable. The nightly matched view is
  [here](out/river-corridor-r2/review/inland-h00-z1.png).
- [New 100-tile forest/jungle region](out/river-corridor-r2/review/freshcanopy-h12-z1.png):
  trunks clear the actual channel, and the sea outlet crosses the old beach
  plug. The [mouth diagnosis](out/river-material-r2/review/mouth-diagnosis.png)
  is explicitly enlarged and compares the old plug, rejected offshore capsule,
  and revised transition. Gameplay comparisons remain unscaled.

The canonical `Renderer/canonical/river.png` and `forest.png` were inspected
beside these results. The reference still has more varied bank exposure and
less uniform pool/river outlines. These are incremental improvements toward
that appearance, not a claim of Civ VI equivalence.

Noon zoom-one changed pixels versus the immediately preceding candidate:

| Region | Canopy layout | River/clearance/material | River candidate source instances |
| --- | ---: | ---: | ---: |
| Coastal | 51,715 | 10,393 | 455 → 428 |
| Inland | 144,050 | 64,500 | 944 → 852 |
| Wilderness | 52,426 | 15,522 | 410 → 401 |
| Fresh canopy | 165,280 | 56,575 | 1,480 → 1,372 |

These counts locate change, not quality. The river candidate removes trunk
placements within the water/bank clearance; it does not thin the whole biome.
Source canopy art, calibrated sizes and individual rotations are preserved.

## Coupled implementation

`systems/objects/canopy_layout.h` permutes existing stratified slots using only
the canonical source-tile seed and stand size. Existing weighted source picks,
jitter, yaw and scale variation are retained. `canopy_variation: 1` opts in.

`systems/hydrology/river_corridor.h` deduplicates all reciprocal BIQ edges using
canonical source identity and the complete supplied halo. It uses the existing
hydrology kernel's coordinate/identity primitives without switching the coast
field. Degree-two endpoints share tangents. A bounded interior bow minimizes
sampled height cost against the actual selected uncarved terrain, retaining a
stable random bow on flat ground. Endpoints and river connectivity stay fixed.

The `river_corridor: 1` adapter connects that single field to water distance,
terrain valley lowering, terminal presentation, expanded receiving-tile
coverage and vegetation clearance in `shared/frozen_scene.cpp`. The river
field uses nominal half-width-64 gameplay pixels; output zoom does not change
the world path. Routing temporarily bypasses river carving to avoid a circular
height query. No new shared hook ABI or native runtime code was added.

Degree-one nodes with all four authoritative neighbors present become small
inland pools or water outlets. Unknown halo boundaries do not become sources.
BIQ flags do not establish flow direction: classifying an inland terminal as a
headwater is a presentation inference. A water terminal extends to an incident
water-cell center, with its material fading into the actual optical shoreline.
This is not simulated hydrology or a gameplay terrain change.

`Q3_CONTINUOUS_RIVERS` uses the existing river base, height, bank noise, clutter
and LEAN channels for source-backed bank/water appearance. It keeps banks out
of the sea while allowing the channel itself to overlap the sea surface.

## Rejected result and remaining gaps

`river-corridor-r1/freshcanopy` is preserved and **rejected** as a combined
visual: extending geometry alone produced conspicuous capped pipes offshore.
`river-material-r2/freshcanopy` was a shader-only diagnosis; its four exact
images match the subsequent full `river-corridor-r2/freshcanopy` composition.

The next three visible fixes are:

1. Banks still read too often as an even outline. Improve their exposed soil,
   erosion and source detail, including dry-bank transition into the ground.
2. Headwater pools are still simple geometric shapes. Direct inspection of the
   existing source-decal BC5 channels shows fine red detail and two irregular
   green footprint-like regions in the top half of the atlas. Their precise
   source-engine meaning is unproven. Investigate a documented source-backed
   adaptation with shared geometry/material coordinates; do not add decorative
   shader masks that disagree with terrain carving.
3. River rocks still use the older edge placement routine. Place source bank
   clutter from the new corridor and inspect remaining steep relief conflicts.
   The current fixed endpoints and bounded bows do not prove every mountain
   configuration is resolved.

Complete the remaining long-coast, relief, volcano and wrap coverage, plus
dawn/dusk, before proposing a broader combined acceptance checkpoint. The new
fresh-canopy region is now a regression witness, not an untuned future region.

## Evidence and reproduction

- `qa/canopy_variation_pass.py --region <region>` prepares/renders stable canopy
  layouts; `--baseline --region freshcanopy` preserves its new baseline.
- `qa/river_corridor_pass.py --region <region> --revision r2` prepares/renders
  the combined candidate. Completed outputs are never overwritten.
- `qa/inspect_canopy_variation.py` verifies 16 frame pairs, fixed inputs and
  unchanged source meshes, texture bindings, counts, sizes and rotations.
- `qa/inspect_river_corridor.py` verifies 16 combined pairs and that the mouth
  shader-only improvement survives composition. Inspectors require Pillow.
- `tests/hydrology/test_river_corridor.py` checks permutation coverage, stable
  canonical curves across shifted source crops, shared endpoints, connected
  pools and height-aware route selection. All four hydrology tests pass.
- Actual selected source terrain was queried at 3,888 matching locations in
  `[76,58]` and `[78,60]` BIQ crops: zero height difference; maximum normal
  component difference `0.000002206`. Recipes are in
  `fixtures/beauty/river-crop-r1`; reports are in `out/river-crop-r1`.
- `renderer_dev.py lab` passes 132 Python tests, 12 Node tests and campaign
  validation. Disabled candidate switches reproduce the two retained coastal
  noon images exactly. No injected compilation or Windows verification is
  claimed for this Lab change.

Machine-readable records: [canopy](CANOPY_VARIATION_r1_EVIDENCE.json),
[combined rivers](RIVER_CORRIDOR_r2_EVIDENCE.json),
[actual source crop check](RIVER_CROP_r1_EVIDENCE.json).

The integration preparation archive is still the old terrain/lighting package.
Its strict verifier now reports expected live-worktree source drift; never
repin it to absorb these candidates. The Integration agent was informed
directly to use that archive and preserve active native/performance work.
