# Single-era city growth hierarchy

The r51/r53-r57 combined candidates improve the distinction between small,
medium and large modern cities. The four-body stage no longer starts with the
tallest compound. Earlier bodies retain their asset, scale, rotation and position
as the city grows. This is a provisional local visual improvement, not approval
of general city quality or a milestone promotion.

[Gameplay-size growth sequence](out/city-growth-hierarchy-r1/inland-growth-native.png)
and [matched previous/candidate comparison](out/city-growth-hierarchy-r1/previous-native-comparison.png).
The crops retain native pixels from 1360x800 renders. Noon and midnight use the
same 100-tile test.biq terrain, camera and non-city placements. Cities remain
explicit Lab augmentations. The existing coastal capital is preserved unchanged.

## Visible result

The old small wilderness city inherited a 98-pixel tower and looked like a
miniature metropolis. The selected four-body city tops out at 38 pixels above
source ground; medium reaches 46 and large reaches 98. These heights come from
choosing different source bodies, not shrinking their proportions. The ordinary
uniform source scale stays 2.9177169657033994 at all stages.

The medium wilderness city gains a lower stepped silhouette. Applying the same
growth recipe to freshshadow reduces its congested central tower wall. That
region had previous city evidence but received no local tuning in this pass.
The inland large arrangement supports the preserved smaller prefixes; it is not
claimed to be universally better than the old large layout in isolation.

Compared with `Renderer/canonical/nightlights.jpg` and the user's Civ VI city
reference, the clearer low-to-tall hierarchy moves toward readable neighborhoods.
Window emission, local spill, source paving and shared shadows survive composition.
The three largest remaining visible gaps are uniform daylight facades, too little
low-building/roofline variety, and sparse detail between building bases. The
reference's historical-era mixing is deliberately excluded by user preference.
Connecting roads remain deferred.

Matched city/shadow/glow changes above 2/255, day / night:

| Case | Pixels |
| --- | ---: |
| Wilderness small | 7,329 / 7,685 |
| Wilderness medium | 7,411 / 7,723 |
| Freshshadow medium | 7,635 / 9,091 |
| Inland large | 13,258 / 13,784 |

Outside the recorded city regions, images are exact except one daylight small
control pixel at 1/255. The original small baseline used a different shadow grid;
the saved `previous-small-fixed-frame` control rebuilds that previous geometry
on the common grid before comparison. The original previous best is retained.

The city-only reflection-off control for the new wilderness medium changes only
three day and two night pixels above 2/255 (maxima 3 and 5). This is a weak
reflection witness, not a visible reflection improvement. The existing capital
lake control remains the stronger evidence. Much of the mirrored geometry may
project onto land in this view; that is an inference, not a completed intersection
analysis. Do not increase global reflection gain to compensate for this view.

## Implementation and checks

`city_growth_layout.solve` now accepts optional per-slot extents. The Lab CLI
`--growth-stage-extents .65 .8 .95` keeps early bodies inside smaller envelopes
while solving the complete planned stage. Omitting it retains the old behavior.
The existing `--graduated-growth` selects lower bodies before taller ones and a
late compound. This is an authored generic adapter, not recovered source-engine
placement. `city_shadow_frame_control.py --shader-render` can retain a previously
composed material/light/paving shader closure during a frame control.

Selected cases: r53/r54/r51 at inland [7,4], sizes 0/1/2, plan size 2;
r56/r55 at wilderness [6,6], sizes 0/1, plan size 1; r57 at freshshadow [5,3],
size 1, plan size 1. All use factor 1.5, source-ground zero, source normals,
AO UV1, emission UV2/gain8/glow, opacity coverage, forest margin .12, facade
spill gain4 and the prior modern settlement-ground recipe. Generator-profile
mixing and the rejected direct-only metalness diagnostic are disabled.

Seven focused layout tests pass. The saved evidence verifies exact growth
prefixes, stage envelopes, unchanged scale, source BIQ/camera identity, terrain
and vegetation clearance, twelve ground-packet isolation checks, twelve fixed
shadow-frame checks and two previous-small controls. All twelve selected Windows
day/night comparisons pass at gameplay zoom1 with current packet/shader hashes.
No new reduced-zoom, full-workflow, native or injected verification is claimed.

Recheck saved evidence with the Python environment providing NumPy/Pillow:

```sh
python3 Renderer/terrain_lab/v2/qa/city_growth_hierarchy_evidence.py
python3 -m unittest discover -s Renderer/terrain_lab/v2/qa -p test_city_growth_layout.py
```

The r50 unrestricted layout is retained as an unrendered diagnostic. r52's
eleven-body wilderness search exhausted 20,000 nodes. This does not prove the
site impossible; a different placement/palette strategy is needed before larger
budgets. The medieval settlement ground remains rejected, with its old best kept.

Next broaden single-era culture/era/size coverage and capital composition using
the existing 47-root palace pack, while investigating facade environment response
and source palette variety. Capital state remains authoritative Civ III input;
style IDs remain generic. Gran Colombian required trees remain unresolved.
No new pack duplication, native changes, frozen pickup changes, human approval
or gate advancement is part of this pass.
