# River bank rock placement

2026-09-06. Small retained improvement within the active river/vegetation
campaign, not acceptance of the full objective. Candidate `river-corridor-r3`
adds only opt-in source rock placement to `river-corridor-r2`.

The original river-rock routine placed source bodies beside the old straight
tile edges. The new river curves could leave those bodies in the channel or
on unrelated high ground. In the inland benchmark, eight original placements
included rocks with ground heights 16.51 and 27.16 source authoring units.

The new `river_bank_rocks: 1` option queries the actual corridor's nearest
segment, offsets in the gameplay pixel metric, and converts back to the terrain
corner lattice. It checks both bank sides with a bounded outward retry. The
posed source footprint must clear the channel/pool core; receiving terrain must
be land, away from the optical sea edge and below the local height threshold.
The existing sparse selection, asset choice, scale and rotation are retained.
Unsafe placements are omitted. No additional decorative rock assets were made.

Visible differences:

- [Inland, actual gameplay size](out/river-corridor-r3/review/inland-h12-z1.png):
  misplaced rocks beside the mountain-foot pool disappear, and remaining rocks
  sit alongside the revised channel. Six of eight placements remain. The
  [night view](out/river-corridor-r3/review/inland-h00-z1.png) retains readability.
- [Jungle outlet](out/river-corridor-r3/review/freshcanopy-h12-z1.png): the small
  rock previously inside the blue channel moves onto its bank. Three source
  bodies remain, with the same art and sizes.
- Coastal: two source bodies remain, with a small bank adjustment. Wilderness
  has no river rocks and is pixel-identical to r2 in all four frames.

Noon zoom-one differences are 110 pixels at the coast, 831 inland and 409 in
the jungle witness. This is a modest placement correction. It does **not**
resolve the much more prominent uniform bank outline or simple source-pool
shape, and must not be presented as a major overall quality jump.

`qa/inspect_river_bank_rocks.py` verifies 16 before/after images, identical
source textures and shader hashes, and exact preservation of all non-rock
source instances. Its [evidence](RIVER_BANK_ROCK_r3_EVIDENCE.json) records each
rock's before/after anchor and grounding height plus pixel bounds. The
hydrology tests now also check bank-query stability across shifted crops and
the pixel-to-world offset conversion; all four tests pass. `renderer_dev.py
lab` passes its 132 Python tests, 12 Node tests and campaign validation.

Reproduce with `qa/river_corridor_pass.py --region <region> --revision r3`.
Completed outputs are preserved. The three fixed regions and fresh-canopy
witness have noon/midnight and both matched zooms. This pass adds no new region
coverage. Native/injected changes belonging to the Integration task were not
edited or tested by this Lab pass.

Next work remains: less uniform exposed river banks, source-backed irregular
pool shapes with aligned geometry/material coordinates, and the outstanding
long-coast/relief/volcano/wrap plus dawn/dusk matrix. The broader goal stays
active, all milestone gates remain unchanged, and the frozen terrain/lighting
integration archive is not updated with these candidates.
