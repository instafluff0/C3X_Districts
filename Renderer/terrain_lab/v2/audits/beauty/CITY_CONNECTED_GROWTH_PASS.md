# Connected city growth and river clearance

The Asian medieval large city now forms a continuous neighborhood instead of
scattering its last eight houses into detached groups. The earlier sixteen
instances remain exact, including asset, scale, rotation and translation. The
bounding area falls from 2.934 to 2.214 square tiles (24.5%). This is a local
composition gain toward the coherent neighborhoods in the canonical city/night
references, while retaining the user's single-era policy and Civ III anchors.

[Gameplay-size day/night comparison](out/city-connected-growth-r1/large-full-light-native.png):
previous large city at left, connected large city in the middle, new ancient-brick
large coverage at right. The two Asian cases use the same 24 bodies, full 80-light
set, materials, camera, terrain and shadow grid. Changed pixels concentrate in
the former gaps and detached outer groups: 8,234 day pixels and 13,088 night
pixels exceed 2/255. Outside the recorded city/shadow/glow region, pixels are
exact. Bounding area alone is not acceptance; the visible neighborhood continuity
is the reason for selecting r65 provisionally.

The compact holdout trials exposed a missing constraint. The old surface query
checked base water, signed shore distance, height and forests, but did not expose
the rendered river corridor. The old r64 medium already overlaps the river;
r68/r69 worsen that problem. These remain preserved diagnostics and are
superseded as clearance candidates. Their earlier material/lighting comparisons
remain valid for those fixed geometries, not proof of valid placement.

[Matched eight-house correction](out/city-connected-growth-r1/river-clearance-native.png):
r69 at left, r72 at right. The right image reveals the complete river bend and
moves the same eight source bodies to connected dry ground without shrinking
them or changing the river/forest. The visible gain changes 4,612 day and 6,528
night pixels above 2/255, with exact outside-city pixels. Both use all 25 local
light proxies. The corrected footprint is slightly wider; clearing the channel
takes priority over minimum bounding area.

## Placement and exclusion contracts

`systems/objects/city_growth_layout.py` adds an optional neighbor-gap constraint.
At `.08` city-local tiles, every completed 8/16/24 growth prefix must form a
connected graph of footprint boxes. Connections can pass through another house;
they need not all touch the center. Candidate ranking prefers a smaller combined
envelope and nearby houses, preserving the fixed source proportions and bounded
search. This gap and ranking are authored C3X presentation choices, not recovered
Civ VI generator parameters. The original unconstrained recipe remains available
for preserved replays.

`--river-clearance-pixels 12.4` in `qa/city_scene_pass.py` exports the actual
terrain packet's interpolated river-distance field before placement. The current
continuous river shader has a maximum water half-width of 6.2 pixels and bank
half-width of 12.4 pixels. Triangle clipping at 12.4 conservatively preserves the
entire authored bank, plus the existing `.012` body padding. This is a Lab adapter
to the current rendered corridor, including its meanders and terminal pools;
it neither changes river geometry nor substitutes a straight tile-edge line.

`qa/river_city_exclusion.cpp` recognizes the actual legacy vertex attribute
profile. Those packets contain world positions at attribute 16 even though
`world_attribute` is unset; filtering only by that semantic field would silently
miss rivers. Unsupported profiles fail. Export records include matched draw and
river-triangle counts, anchor, packet hash and threshold. The generic
`systems/objects/city_exclusion.py` consumes convex XY polygons, using a spatial
index and exact rectangle/convex-polygon intersection rather than five point
samples. The evidence script independently checks overlap by rectangle clipping.
Production should provide generic exclusion geometry from its authoritative
terrain/river service; it need not parse this Lab packet profile or source art.
Connecting roads remain deferred.

Use the river exclusion for future city placement, including palace studies.
Do not preserve an earlier footprint that already violates it. A failed fit is
not permission to shrink bodies, cover water, clear forests or alter the fixed
terrain. Existing captured-capital authority and style fallback contracts remain
unchanged.

## Coverage and preserved failures

| Revision | Region / city | Result |
| --- | --- | --- |
| r65 | Inland Asian medieval, 24 houses | Connected; earlier 16 exact; selected local improvement |
| r66 | River holdout, 24 houses preserving r64 | 20,000-node budget exhausted; preserved failure |
| r67 | River holdout, fresh 24-house plan | 20,000-node budget exhausted; preserved failure |
| r68/r69 | River holdout, connected 16/8 | Rendered; rejected for river overlap |
| r70 | Inland ancient-brick, 24 houses | Connected; earlier 16 exact; new large-size coverage |
| r71 | River holdout, 16 with bank exclusion | 20,000-node budget exhausted; not an infeasibility proof |
| r72 | River holdout, 8 with bank exclusion | Fits in 18,593 nodes; connected and clear; selected correction |
| r73 | Previously city-untuned coast, ancient-brick 16 | Fits in 16 nodes, zero backtracks, unchanged recipe |

The separate r73 region is the preserved `freshwater` 100-tile terrain window at
BIQ origin `[52,30]`; the name is a fixture identifier, not a freshwater-waterbody
classification. The city anchor `[8,3]` was chosen from flat coastal terrain
before city rendering. No palette, scale, gap or local search tuning was applied.
[Native-size coastal day/night witness](out/city-connected-growth-r1/untuned-coast-native.png).
It retains a low single-era skyline and warm windows on the coast. This is a
generalization witness, not a matched prior-city improvement. No new substantial
reflection-quality claim is made; the existing American capital/lake reflection
witness remains preserved.

All six final comparison/coverage scenes use the buffered-light composition
after shared shadows. Twelve Metal/D3D gameplay-size comparisons pass. Twelve
independent terrain checks preserve source geometry, material/texture bytes,
constants and existing object placement; feature attributes receive only zero
padding. Twelve light-buffer checks and ten fixed-shadow-frame checks pass.
Fifteen focused layout/exclusion tests pass. These support the rendered result;
they do not provide human visual approval. The new coastal parity case has a
higher mean channel difference (0.254/0.279 out of 255) than the inland cases,
with unchanged silhouette and a passing existing gate.

Recheck saved evidence with Python providing NumPy/Pillow:

```sh
python3 Renderer/terrain_lab/v2/qa/city_connected_growth_evidence.py
python3 -m unittest discover -s Renderer/terrain_lab/v2/qa -p test_city_growth_layout.py
python3 -m unittest discover -s Renderer/terrain_lab/v2/qa -p test_city_exclusion.py
```

`CITY_CONNECTED_GROWTH_EVIDENCE.json` retains exact hashes, search outcomes,
growth-prefix checks, clearance and matched pixel bounds. Re-render from each
preserved buffered case's `report.json`, `combined.hlsl` and `reflection.hlsl`
with its saved postprocess closure. Do not regenerate over an existing revision.
`CITY_CONNECTED_GROWTH_CLEANUP.json` records removal of 36 completed new linear
readbacks (298.8 MiB); all replay inputs, images, failures and previous bests remain.

## Next visible gaps

The canonical `nightlights.jpg` still has clearer architectural landmarks,
more distinct facade/material surfaces and better organized open ground than
these horizontal house groups. Next connect the appropriate single-era palace
styles from the 47-root pack, preserving source proportions and river clearance;
extend culture/era coverage at meaningful combined checkpoints. Medium/large
growth beside the river remains unresolved and needs an approach that reasons
about available connected dry land, not repeated increases to the same search
budget. Source environment response and repetitive small-house roof patterns
also remain open. No overall Civ VI-quality claim, native delivery, milestone
advance or human approval is recorded.
