# Individual-house city density

The Asian medieval and Mediterranean ancient pools expose a problem hidden by
modern towers: a seven-component budget produces only seven small houses. The
imported pools contain individual buildings, not the assembled modern blocks.
Restoring their materials alone does not make them read as cities.

The r60/r61 sixteen-house studies occupy a recognizable neighborhood at the same
source scale as the r58/r59 seven-house controls. Their rooflines, courtyards and
facades vary using existing source assets. The ancient pool is the source ancient
brick family; its pool name does not establish uniquely Mediterranean architecture.
One era is used throughout each city. No roads or new ground atlas patches are
introduced. The rejected broad medieval paving extension stays rejected.

## Scene and implementation

All scenes preserve the 1360x800 gameplay camera and 100-tile test.biq terrain.
Inland uses anchor [7,4] and the prior r48 shadow grid. Freshshadow uses [5,3] and
the r49 grid. Cities are explicit Lab augmentations, not captured BIQ city state.
Source UVs, source-ground zero, frame normals, AO UV1, emission UV2/gain8/glow,
gloss and source ground triangles are retained. The new source-material overlays
contain no extra opacity or metalness textures; no missing channels are invented.

`city_scene_pass.py --stage-component-counts 8 16 24` supplies explicit bounded
budgets to the constrained solver. The existing 4/7/11 default is unchanged.
This option requires constrained growth and increasing counts of at most 32.
The default small/medium/large envelopes remain .65/.8/.95 tiles, with .12 forest
clearance and factor 1.5. There is no runtime culture-name special case.

The medium and large inland Asian candidates solve the same 24-body plan; the
first sixteen placements are exact. However, the large r62 result scatters its
outer houses and is **not selected as a visual improvement**. It is a capacity
and composition witness. A future compactness/connected-neighborhood placement
approach is preferable to adding more houses to the same search objective.

The holdout's r63 future 24-body plan exhausts 20,000 nodes. r64 uses the same
palette, scale, density, lighting and envelopes but plans only its current
16-body stage, which fits in sixteen nodes without backtracking. This verifies
the current stage, not stable growth to a future large city on that site. The
region had earlier city studies; it receives no culture-specific site tuning.

## Light composition

The original facade probe allowed twelve bodies. Simply increasing its limit
produced 53 proxies for the medium Asian city and 80 for the large one. The
53-proxy Metal trial remained inside an active compiler for several minutes
after the small controls completed. That trial was deliberately stopped; its
unfinished source closure remains under `out/city-culture-r1/asian-medium`.
Do not report it as a rendered comparison or a runtime GPU failure.

The bounded adapter retains the strongest spill proxy for every emitting body,
then adds the strongest remaining proxies until the requested budget is reached.
The studied `--light-budget 32` keeps all window emission, geometry, reflection
geometry and HDR glow intact. Only the authored secondary spill approximation is
reduced. Proxy order remains stable; source energy ranking uses linear luminance,
intensity and squared range. This is not recovered source light placement or
energy-conserving area-light transport. Full building blockers remain present.

Candidate composition is under `out/city-culture-r2/`. The previous seven-house
controls use the same lighting recipe. The bounded 32-proxy Metal attempt also
spent several minutes actively compiling and was stopped. Reducing proxy count
does not resolve the underlying compilation problem. The next implementation
step should move light/blocker data out of large shader-embedded constant arrays
and into generic runtime data, with matched pixel controls. Do not repeatedly
restart the same expensive compiler trial or discard source window emission.

`replay_shader.py --prepare-only` now publishes the full frozen shader closure
and batch without reporting a render pass. The postprocess source is published
before GPU execution. `city_d3d_probe.py --render-only` produces Windows images
with `metrics: null`; ordinary `--resume` later checks the recorded input/output
hashes and fills parity metrics only when Metal images exist. An initial Windows
attempt exposed the late publication of the postprocess source; that interface
is corrected. A subsequent transient VM dispatch error was resumed without
replacing any completed frame.

The visual checkpoint uses matched **Windows** frames throughout. Metal
comparisons for the denser candidates remain pending; no cross-backend or native
acceptance is inferred from successful Windows rendering.

All twelve Windows day/night frames completed. Four seven-house baseline Metal
comparisons pass; eight denser-frame comparisons remain pending. Twelve focused
tests (eight layout, four light sampling/budget) pass. The saved evidence checks
source-pool membership, source scale, medium/large prefix stability, dry/forest
clearance, twelve fixed shadow frames and unchanged geometry/emission packets
through lighting composition. Every emitting body retains a spill proxy.

[Matched culture comparison](out/city-culture-r2/selected-native-comparison.png)
and [large/holdout witness](out/city-culture-r2/large-holdout-native.png) retain
native pixels. Asian medium changes 6,810 day / 8,865 night pixels above 2/255;
ancient medium changes 6,632 / 9,985. All pixels outside the recorded city regions
are exact. These counts locate the changed neighborhood and lights; they are
not a similarity score or broad visual approval. Source intake adds 2.9 MB of
normal/frame and ground mappings, sharing the existing asset pack and textures.

## Review and next work

At gameplay size, the denser medium cities move toward the distinct neighboring
roof volumes visible in the user's Civ VI city example. The canonical
`Renderer/canonical/nightlights.jpg` remains the warm-light/readability reference;
its mixed eras are excluded by user preference. The three largest remaining
visual gaps are overly dark/flat exterior surfaces, sparse detail between houses,
and scattered growth at the large stage. A new culture image is not automatically
a visual improvement; compare the saved paired controls.

The prior American palace and strong capital-lake reflection control remain
preserved. This pass does not broaden palace selection or establish new shoreline
reflection quality. Other culture/era/size combinations, new palace composition,
native implementation and every approval gate remain open.

Recheck saved composition evidence with Python providing NumPy/Pillow:

```sh
python3 Renderer/terrain_lab/v2/qa/city_culture_evidence.py
python3 -m unittest discover -s Renderer/terrain_lab/v2/qa -p test_city_growth_layout.py
python3 -m unittest discover -s Renderer/terrain_lab/v2/qa -p test_city_facade_light_sampling.py
```

Subsequent resolution: [buffered-light data](CITY_LIGHT_BUFFER_PASS.md) supersedes
the eight pending dense-frame comparisons with passing Metal/D3D equivalents and
exact old Windows images. The historical static-array trial records remain
unchanged; do not restart them. Full 53-proxy spill also yields a local night
improvement in the two Asian medium scenes.
