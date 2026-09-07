# Combined terrain checkpoint — larger source relief

2026-09-06. Sole lead, local Metal. Retained work in progress:
**relief-size-r3**, including **combinedvolcano** as a separate synthetic witness.
The three fixed 100-tile test.biq benchmarks, 128-tile long coast, previous fresh
coast, and a newly selected 100-tile region with four mountains were composed
at noon/midnight and both fixed gameplay zooms. The synthetic volcano adds
four frames to those 24 real-map frames.

This is an incremental visual improvement, not Civ VI-level acceptance.
No human approval, milestone closure or Integration promotion is recorded.
Cities, units and improvements remain deferred.

- [Larger inland mountains — native gameplay comparison](out/relief-size-r3/review/inland-day-z1-comparison.png)
- [Larger volcano in the current combined scene — explicitly synthetic](out/relief-size-r3/review/combinedvolcano-day-z1-comparison.png)
- [Combined volcano and mountains at night](out/relief-size-r3/review/combinedvolcano-night-z1-comparison.png)
- [Inland mountains at night](out/relief-size-r3/review/inland-night-z1-comparison.png)
- [Fixed coast — full native zoom 2](out/relief-size-r3/review/coastal-day-z2-comparison.png)
- [Wilderness — full native zoom 2](out/relief-size-r3/review/wilderness-day-z2-comparison.png)
- [Long coast — full native zoom 2](out/relief-size-r3/review/longcoast-day-z2-comparison.png)
- [Previously untuned four-mountain region](out/relief-size-r3/review/freshrelief-day-z2-comparison.png)
- [Previously untuned region at night](out/relief-size-r3/review/freshrelief-night-z2-comparison.png)
- [Full combined inland scene](out/relief-size-r3/inland/h12-z1-pan00.png)
- [Full combined volcano witness](out/relief-size-r3/combinedvolcano/h12-z1-pan00.png)

Mountains use 1.30 uniform source-body scale and volcanoes 1.60 before the
bounded topology joins. Foothills can extend .25 tile beyond an ownership edge,
ending before the neighboring center. Source height samples, fixed cameras and
terrain remain unchanged. Vegetation keeps its XY placement, mesh, scale and
yaw; grounding heights follow the changed terrain. An initial larger volcano
exposed a tile-bound material cut. That version was rejected; the retained
version carries source-owner coordinates and coverage across the skirt.

The changed footprints give the terrain more presence beside forests and move
toward canonical `mountain.png`. Detailed source/adaptation limits, actual-pixel
locations and validation are in [RELIEF_SIZE_PASS.md](RELIEF_SIZE_PASS.md).
The previous best `coast-pass-rocks-r8` and every intermediate render are preserved.

The three largest remaining gaps are:

1. Mountain/volcano surface projection and natural shoulder integration. The
   volcano remains streaked; source physical reconstruction is still unproven.
2. Shadow-receiver lines across dunes, plus the inherited unapproved analytic
   dune body. Disabling all shadows isolates the lines, but is not a retained fix.
3. Soft shallow-water structure and abrupt cliff/grass joins.

[RELIEF_SIZE_r3_EVIDENCE.json](RELIEF_SIZE_r3_EVIDENCE.json) verifies all 28
matched frames and preserved source/placement inputs; it does not decide quality.
[RELIEF_SHADOW_DIAGNOSTICS.json](RELIEF_SHADOW_DIAGNOSTICS.json) records the
unselected lighting isolation. The new four-mountain region is now a regression
witness, not an untuned witness for later acceptance. Earlier retained coastal
and water work is documented in [COAST_SOURCE_JOIN_PASS.md](COAST_SOURCE_JOIN_PASS.md).
LQ0 remains ready/unaccepted.
