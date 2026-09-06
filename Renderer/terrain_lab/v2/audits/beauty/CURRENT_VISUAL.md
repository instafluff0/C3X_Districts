# Combined terrain checkpoint — varied coast and volcano witness

2026-09-06. Sole lead, local Metal. Current retained work in progress:
**coast-pass-rocks-r4**, with **volcano-witness-r2** as a separate synthetic
supplement. All three fixed 100-tile real-map regions and the new 128-tile long
coast were rendered at noon/midnight and both gameplay zooms. Matched crops,
full scenes and the native zoom-2 matrices were directly inspected.

This is an incremental visual improvement, not Civ VI-level acceptance.
No human approval, milestone closure, Integration promotion or injected change
belongs to this pass. Cities, units and improvements remain deferred.

- [Long shoreline before/after — native gameplay pixels](out/coast-pass-rocks-r4/review/longcoast-day-comparison.png)
- [Fixed coast before/after](out/coast-pass-rocks-r4/review/coastal-day-comparison.png)
- [Inland before/after](out/coast-pass-rocks-r4/review/inland-day-comparison.png)
- [Wilderness before/after](out/coast-pass-rocks-r4/review/wilderness-day-comparison.png)
- [Matched long-coast night comparison](out/coast-pass-rocks-r4/review/longcoast-night-comparison.png)
- [Supplemental volcano mapping before/after](out/coast-pass-rocks-r4/review/synthetic-volcano-day-comparison.png)
- [All regions at native zoom 2, day](out/coast-pass-rocks-r4/review/all-regions-z2-day.png)
- [All regions at native zoom 2, night](out/coast-pass-rocks-r4/review/all-regions-z2-night.png)
- [Canonical Civ VI detail and Civ III shoreline references](out/coast-pass-rocks-r4/review/canonical-shore-references.png)
- [Full 128-tile long coastline](out/coast-pass-rocks-r4/longcoast/h12-z1-pan00.png)

The new headlands and coves replace the previous smoother outline. Source cliff
bodies distinguish rocky hill edges from sandy lowlands. Direct source hill
sampling restores lower relief erased by the old threshold and makes its UVs
stable across crops. The supplemental volcano has a broader, clearer crater;
its height, dormant/active color, slope and specular channels now share one
coordinate mapping. Detailed findings and limitations are in
[COAST_VOLCANO_PASS.md](COAST_VOLCANO_PASS.md).

The three largest visible gaps remain:

1. Rock-to-hill composition: source bodies still read as separate brown boulders
   or rings around ponds. They do not yet form the reference's continuous,
   exposed cliff faces. Further random scale changes are not the remedy;
   investigate the coastal relief envelope, grass shoulder and material response.
2. Coast material structure: broad soft shallows and weak submerged detail still
   hide fine shoreline character. The terrain/water join must remain seamless.
3. Relief appearance: inherited analytic dunes remain diagnostic proxies with
   thin seams; mountain/volcano proportions and side texture projection still
   fall short. Source height hashes do not establish source physical geometry.

`candidate-v5` remains the preserved previous best. Its earlier improvements,
rejected dune/coast experiments and evidence are archived in
[GAMEPLAY_TERRAIN_PASS.md](GAMEPLAY_TERRAIN_PASS.md). New source/fixture hashes,
matched-camera checks, actual draw-channel bindings and changed-pixel locations
are in [COAST_rocks-r4_EVIDENCE.json](COAST_rocks-r4_EVIDENCE.json). These support
inspection; they do not decide visual acceptance. LQ0 remains ready/unaccepted.
