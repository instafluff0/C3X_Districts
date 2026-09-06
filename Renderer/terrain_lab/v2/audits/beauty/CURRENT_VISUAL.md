# Combined terrain checkpoint — selected source cliffs and water

2026-09-06. Sole lead, local Metal. Retained work in progress:
**coast-pass-rocks-r8**, with **volcano-witness-r2** as a separate synthetic
supplement. All three fixed 100-tile test.biq regions, the 128-tile long coast,
and a previously untuned 100-tile coast were rendered at noon/midnight and both
fixed gameplay zooms. Native crops and full zoom-2 scenes were directly inspected.

This is an incremental visual improvement, not Civ VI-level acceptance.
No human approval, milestone closure or Integration promotion is recorded.
Cities, units and improvements remain deferred.

- [Long coastline before/after — unchanged gameplay pixels](out/coast-pass-rocks-r8/review/longcoast-day-comparison.png)
- [Fixed coastal benchmark before/after](out/coast-pass-rocks-r8/review/coastal-day-comparison.png)
- [Inland before/after](out/coast-pass-rocks-r8/review/inland-day-comparison.png)
- [Wilderness before/after](out/coast-pass-rocks-r8/review/wilderness-day-comparison.png)
- [Long coast at night](out/coast-pass-rocks-r8/review/longcoast-night-comparison.png)
- [Previously untuned coast — full native zoom 2](out/coast-pass-rocks-r8/review/freshcoast-z2-day-comparison.png)
- [Previously untuned coast at night](out/coast-pass-rocks-r8/review/freshcoast-z2-night-comparison.png)
- [All fixed regions at native zoom 2, day](out/coast-pass-rocks-r8/review/all-regions-z2-day.png)
- [All fixed regions at native zoom 2, night](out/coast-pass-rocks-r8/review/all-regions-z2-night.png)
- [Canonical Civ VI and Civ III references](out/coast-pass-rocks-r8/review/canonical-shore-references.png)
- [Full 128-tile long coastline](out/coast-pass-rocks-r8/longcoast/h12-z1-pan00.png)
- [Supplemental volcano mapping](out/coast-pass-rocks-r8/review/synthetic-volcano-day-comparison.png)

Gray selected-source cliff materials replace the incorrectly inherited Base
materials. A subordinate grassy shoulder supports overlapping source bodies,
turning several rows of detached boulders into connected rocky banks. The
source water normal/moment textures now affect shared lighting instead of
remaining bound but unused. The retained varied shoreline itself stays fixed.
The first buried-rock join was rejected; all attempts and previous best r4
remain preserved. Source controls, adaptation limits, changed-pixel locations,
and checks are recorded in [COAST_SOURCE_JOIN_PASS.md](COAST_SOURCE_JOIN_PASS.md).

The three largest visible gaps remain:

1. Some grass/rock joins are abrupt, and pond banks still lack the reference's
   natural exposed face and shoulder relationship.
2. Soft shallows hide bed structure. Static water normals improve the open
   surface, but shoreline interaction and reference-like water layering remain.
3. Inherited analytic dunes are unapproved diagnostic proxies with seams;
   mountain/volcano physical extent and side projection remain unresolved.

The new coast is now a regression witness, not an untuned witness for future
acceptance. Source terrain and cameras remain pinned in its foundation fixture.
[COAST_rocks-r8_EVIDENCE.json](COAST_rocks-r8_EVIDENCE.json) verifies the matched
20-frame composition and actual material bindings; it does not decide quality.
[COAST_SOURCE_BINDING_DIAGNOSTICS.json](COAST_SOURCE_BINDING_DIAGNOSTICS.json)
separates source material, shoulder, water and rejected geometry experiments.
Earlier retained passes are documented in
[COAST_VOLCANO_PASS.md](COAST_VOLCANO_PASS.md) and
[GAMEPLAY_TERRAIN_PASS.md](GAMEPLAY_TERRAIN_PASS.md). LQ0 remains ready/unaccepted.
