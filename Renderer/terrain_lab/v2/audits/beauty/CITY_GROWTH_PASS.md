# City growth and stable shadow comparison r40–r49

Status: provisional wilderness composition improvement; full city goal, human
review and milestone gates remain open. No native/injected code changes.

## Visible result

The selected local candidate is r46 geometry rendered with the r37 shadow frame:
`out/city-growth-r1/r46-fixed-shadow-frame/render/`. Compared with r37, the same
seven source bodies have legal forest clearance and a more readable skyline:
taller silhouettes sit behind lower roofs, while foreground roof outlines and
facades remain distinct. The source bodies, uniform scale, single modern era,
anchor [6,6], 100-tile wilderness terrain, forest placement, camera, lighting
and output size are unchanged. Individual building translations/orientations
are the intended layout changes. This is a local improvement, not a universal
replacement city recipe.

At normal 1360×800 gameplay output, 8,515 noon and 7,130 midnight pixels differ
by more than 2/255. All changed pixels are around the city and its shadows/glow;
pixels outside [755,245,1000,450] are **exactly unchanged**. Compare
`out/city-growth-r1/selected-native-comparison.png` at native pixel size.
The pre-fix r46 render and r43 crowded skyline remain preserved diagnostics.

The canonical `Renderer/canonical/nightlights.jpg` still shows a stronger civic
ground treatment, richer facade illumination and locally lit surroundings.
Our warm windows remain readable, but this pass does not restore environment
specular, local light pools or improve the water reflection model. Source-ground
paving remains fragmented. The full source screenshot's historical era mix is
not a target: the user explicitly wants one era per city.

## Layout investigation

The previous late-slot greedy retries could never succeed around r39's fixed
four-body core on the tested grid: slot 6 has zero legal candidates, even after
allowing both quarter-turn orientations and testing a larger extent (r40/r41).
This is a finite-grid result, not proof about all continuous arrangements.

`systems/objects/city_growth_layout.py` now offers an opt-in bounded search:
precompute dry/height/vegetation-safe sites, select the most constrained remaining
body, prefer placements that leave more sites available for later bodies, and
backtrack. It retains source geometry/scale and returns instances in growth order.
It can preserve an explicitly supplied prior prefix, or plan a later stage and
render its exact prefix. It never silently substitutes bodies or shrinks them.
An exhausted node budget is distinct from exhausting the tested placement grid.

r43 proved seven-body feasibility but looked too crowded. Reserving room for
later bodies produced the clearer r46 result in 39 search nodes. r47 is the exact
first four buildings of the same seven-body plan, using the same 0.8-tile
half-extent. It does not independently repack the smaller stage. The initial
r39 arrangement remains preserved but is not the prefix of this new plan.

Eleven bodies still did not fit at this wilderness anchor within the bounded
r42/r44/r45 searches, including a finer placement grid. Do not turn these
failures into an infeasibility claim or keep increasing the search budget.
Source component alternatives and skyline-aware placement remain next work.
The same solver does place all eleven source bodies at the fixed inland anchor
[7,4] (r48), with dry ground, relief-height and vegetation-clearance checks.
That is an additional case, not proof of complete three-stage growth everywhere.

r49 is a previously unused **city** region: the existing freshshadow 100-tile
test.biq window at anchor [5,3]. The site was selected from raw terrain cells
before its city render, and no local tuning followed. Its distinct existing
`shadow-receiver-r1` natural benchmark is retained, not relabeled as the later
river benchmark. Clearance passes, but the skyline is crowded. This exposes
the solver's limited visual objective and prevents general visual acceptance.
Freshshadow is now a regression witness rather than an untuned city witness.

## Shadow composition blocker and correction

Moving city geometry changed the all-scene bounds used to center the shadow
map. In the r46/r37 day comparison, its world-Z origin moved from approximately
0.802575 to 0.798531. Refitting the grid changed distant terrain/forest shadow
samples, even though that geometry stayed fixed. Those pixels were not counted
as city improvements.

`systems/lighting/scene_shadow.cpp` accepts an optional reference packet to retain
its light-grid origin, extent and resolution while rasterizing all current
casters. It checks the light direction, frame dimensions and complete current
scene bounds, rejecting a reference that would clip the scene. Existing calls
retain their original fitting behavior. This is an explicit Lab comparison
control, not a completed native shadow-cache/envelope policy.

`qa/city_shadow_frame_control.py` produces the checked matched replay without
altering the earlier packets. `qa/city_scene_pass.py --shadow-frame-report` lets
future layout comparisons use that reference during their initial shadow build.
The actual packet contract verifies unchanged geometry/coverage and identical
light-grid constants. With this control, every pixel outside the city area
matches r37 exactly. Current city shadows are rebuilt, not copied from r37.

## Reproduction and verification

Use the ordinary modern city material arguments from r34/r39, plus:

```text
--region wilderness --anchor 6 6 --factor 1.5 --expanded
--vegetation-clearance .12 --growth-search-nodes 20000
```

For the r47 smaller stage add `--size 0 --footprint-limit .8 --growth-plan-size 1`.
For a new matched wilderness render, supply
`--shadow-frame-report Renderer/terrain_lab/v2/audits/beauty/out/city-scene-r37/american-modern-s1-wilderness-at6-6/report.json`.
`--layout-only` checks plans before any GPU work. `--preserve-layout` takes a
previous augmentation for an immutable existing prefix; it reports a failure
when that prefix prevents the requested growth.

Run `qa/city_growth_evidence.py` with Python providing Pillow/NumPy and clang++
on PATH. [CITY_GROWTH_r40_r49_EVIDENCE.json](CITY_GROWTH_r40_r49_EVIDENCE.json)
records placements, exact growth prefix, actual material/geometry identities,
clearance against frozen samples, failed searches, image deltas and shadow-frame
packet checks. Ten Windows comparisons pass across r46/r47/r48/r49 and the
selected fixed-frame replay. Seventeen focused tests pass, including constrained
packing, quarter-turn fit, immutable-prefix failure and budget classification.
Rebuilding the unchanged r37 input through both the default and explicit-reference
shadow paths reproduces its shadow texture bytes exactly; the no-op packet checks
are recorded in `CITY_GROWTH_SHADOW_NOOP.json`.
These are supporting checks, not visual acceptance. No full-Lab or milestone
completion is claimed.

Disposable linear render buffers from this pass are removed after evidence;
`CITY_GROWTH_CLEANUP.json` records the exact files. Previous images, replay
packets, source packs and unsuccessful search records remain intact.

The three largest open gaps are general city arrangement/ground coverage,
facade environment and local night lighting, and complete large-city growth
across terrain/culture/era configurations. Roads remain deferred. Capital
selection retains the broader imported roster and explicit native-capital
authority contract; no new palace mapping is implied by this layout pass.
