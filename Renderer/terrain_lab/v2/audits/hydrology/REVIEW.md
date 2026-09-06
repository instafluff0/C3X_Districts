> Resumed correction: Q0/Q6 now provide genuine scene-linear per-draw and
> source hydrology-field channels. The legacy mixed-shader reproducer below
> is historical and is **not a current platform blocker**. Source-composed
> material migration is active. The user rejected the large proxy overview's
> appearance; it is not an accepted beauty result.

# Q3 static hydrology candidate 07 — convergence handoff, not acceptance

Q3 now supplies a continuous shoreline and beach envelope, monotone bathymetry,
static source-textured water/bed, canonical river curves, relief-carving inputs,
exact crossing anchors, and river/bank placement exclusions. Implementation and
all evidence remain in Q3-owned paths. Overall acceptance is **pending**.

## Source-composed before/after

The matched Q2/Q6 wet-region render now consumes Q3 through Q0's opt-in
`HydrologyHooksV1`. Direct inspection confirms the large pale rectangular marsh
plate and sharp lip are gone; continuous grass/marsh/source pools appear beneath
the same source vegetation. This is a verified real-map source-composed fix.
`COMPOSED_REGRESSION.json` pins both images and proves matching terrain, source
hash, assets, placement layers, shaders, phase, zoom, viewport and sampling.
Only the Q3 shoreline callback is enabled in the after variant.

![Before: pale plate and lip](out/composed-regression/before.png)
![After: continuous grass and marsh](out/composed-regression/after.png)

The callback fixes shore classification/relief support. It does not yet compose
Q3's full static water material, source cliff joins or river corridor rendering.

## Contextual evidence first

The small open-coast context uses a fixed **128x64-pixel tile basis**, with no
inventory diorama, invented buildings, or fake captured gameplay. It is explicitly
synthetic terrain with diagnostic ground/relief receivers; the material inputs
are normalized alternate-skin source assets. The small black corner is the
bounded diagnostic patch, not a missing in-game tile. Full gameplay composition
requires the shared owner scene rather than decorating this proxy with fake art.

- Current noon: `out/Q1/Q3-hydrology/context-07/h12-z1-pan00.png`.
- Current night: `out/Q1/Q3-hydrology/context-07/h00-z1-pan00.png`.
- Fixed-camera early defect/revision pair: `context-02/h12-z1-pan00.png` and
  `context-04/h12-z1-pan00.png`, under the same output root. This pair shows the
  wet-line brightness correction and removal of sand beneath the rocky proxy.
- Matched current material ablation: `context-before-static-06/h12-z1-pan00.png`
  versus `context-07/h12-z1-pan00.png`. The ablation forces uniform beach width
  and sandy treatment; it is **not** a frozen v1 reference or claimed game capture.
  Both use the same source materials, terrain, camera and pixel scale.

The final context has 32 deterministic variants: four phases, both zooms and
four camera offsets including return to origin. All eight phase/zoom returns
are byte-identical. Mean displayed luminance is 0.4437 noon, 0.2804 sunset,
0.1934 midnight and 0.2803 sunrise. Noon luminance SD is 0.0792; mean horizontal
luminance difference is 0.00521. These describe contrast and phase ordering;
they do not numerically prove beauty. Per-frame costs and effective settings
remain beside each output. The contextual noon diagnostic used about 38 MB of
reported GPU allocations and about 2.3 ms render-only GPU time in one sampled
run; this is not a complete-game performance estimate.

## Direct visual review and revisions

Review: Codex/GPT-6 direct image inspection, Q3 rubric v1, moderate confidence
for material appearance and high confidence for the directly witnessed geometry
failures. Canonical `sea_and_shore.png` and `river.png` were inspected directly.
They are property targets, not pixel-comparison baselines. White captured wave
crests are deferred transient effects, not an instruction to paint a foam band.

1. First coast: the water/land join was connected, but the beach looked uniform
   and the bed too soft. Source shallow-height material response was added.
2. The first cove fixture accidentally used corner-only water connectivity. A
   true edge-connected cove was recorded; corner-touch islands remain a stress
   test rather than being called a connected cove.
3. Numeric crossings exposed an edge-ID collision that dropped separate river
   segments. Ordered raw-corner FNV IDs replaced the lossy XOR combination;
   reciprocal descriptions still deduplicate to exactly one physical curve.
4. River images showed pale channels and raised proxy valley walls obscuring
   their continuity. The water tint was restrained toward gray-blue, node
   tangents joined degree-two bends, and a wider carving envelope removed the
   visual cuts. Source topology and endpoint positions did not move.
5. Bed-only comparison exposed wet sand becoming brighter below the waterline.
   Wet material now continues under water without the permanent pale seam.
   Fully rocky edges use rock-to-shallow-bed material rather than hidden sand.
6. The isolated islands revealed a corner-contact bridge; bounded shoreline
   erosion now separates that connection while preserving the land centers.
7. Classification isolation revealed stepped rocky fractions from contour
   segment ownership. Final sampling evaluates continuously at the contour
   foot; the red/green diagnostic now has a smooth transition.
8. Camera-independent material UVs are world-anchored and use integral repeat
   periods across raw horizontal wrap. Hydrology has no animated normals,
   waves, foam, caustics, flow inference or time-driven redraw request.

Inspected diagnostics include coves, separated islands, a narrow channel,
long shoreline runs, all four boundary orientations, fully sandy/rocky controls,
mixed corners, river sources/joins/mouths, bed-only, beach-only, classification,
and clearance overlays. Both-zoom/four-phase contact sheets were inspected for
these cases in revision 06, and the changed cove/island/channel cases were
inspected again in revision 07. No periodic S-template or painted surf ribbon is
used. River channels remain thin at reduced zoom; shallow-bed detail remains
subtle and must be checked with Q1's final sampling policy. The generated smooth
rocky receiver is visibly too regular to represent the desired source cliff;
it is explicitly a **diagnostic proxy**, not an approved rock solution.

## Verified real terrain

Source `test.biq`: SHA-256
`a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e`,
17173 bytes, 100x100 raw map, horizontal wrap. Q0's registry and cached adapter
are consumed read-only. CSV terrain and river masks are unchanged; no Lab object
augmentation is present. Q3 explicitly exports halo **2**, independent of Q0's
larger default for frozen shadow receivers.

| Region | Raw origin | Evidence |
| --- | --- | --- |
| mixed | (14,42) | Rocky hill/lowland coastal context; revision 07 both zooms/four phases |
| mixed-holdout | (18,46) | Fixed river neighbor; revision 07 both zooms/four phases |
| wrap | (94,50) | Coastal wrap context; revision 07 both zooms/four phases |
| wrap-holdout | (98,54) | Fixed all-ocean neighbor; revision 07 both zooms/four phases |
| q3-mouth | (85,21) | Ten river tiles and four visible land/water edges; revision 07 both zooms/four phases |
| q3-mouth-holdout | (89,25) | Fixed coastal neighbor; revision 06 both zooms/four phases; final rerun pending |

All six regions were rendered and directly inspected. These are genuine
source-terrain witnesses but **not** full source-feature beauty acceptance:
forest, jungle, relief and terrain-family receivers are still represented by
Q3's clearly labeled ground/relief proxy. `real_coverage.json` records broad
coverage and the read-only mouth-region search that Q0 registered. It does not
invent flow direction or declare undocumented fine topology present.

## Executable verification and reproduced blocker

From the repository root:

```sh
python3 -B -m unittest discover -s Renderer/terrain_lab/v2/tests/hydrology -p 'test_*.py'
python3 Renderer/terrain_lab/v2/fixtures/hydrology/prepare_real.py
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/hydrology/context.fixture.json --candidate review-next
python3 Renderer/terrain_lab/v2/app/runner.py check --fixture Renderer/terrain_lab/v2/fixtures/hydrology/context.fixture.json --settings Renderer/terrain_lab/v2/fixtures/hydrology/scroll.settings.json --candidate review-scroll
python3 Renderer/terrain_lab/v2/systems/hydrology/publish_corridors.py Renderer/terrain_lab/v2/fixtures/hydrology/rivers.fixture.json Renderer/terrain_lab/v2/fixtures/hydrology/rivers.corridors.json
python3 Renderer/terrain_lab/v2/app/runner.py validate --fixture Renderer/terrain_lab/v2/fixtures/hydrology/mixed-shader-blocker.fixture.json
```

The last command intentionally reproduces:
`mixed shader contracts require a versioned per-draw shader interface`.
Q0 must expose per-draw shader identity and the Q6 scene-linear shared display
boundary before the full Q3 static material can compose with source terrain/relief/networks.
The separate signed-shore hook now works in Q2's Q6 scene-linear branch; that
partial composition does not remove the mixed-packet shader limitation.
The current local shader encodes its own complete opaque diagnostic once;
stacking it after another owner's tone map would be incorrect.

The portable tests prove shoreline class/continuity, monotone normal-ray depth,
reciprocal river deduplication, exact/reversed crossings, missing-halo rejection,
raw-wrap identity, endpoint/carving invariants, full-footprint clearance, and
conservative shared polygon coverage. Two Python tests exercise the C++ kernel
and Q0's real corridor schema/predicate; all pass. JSON envelopes and numeric
crossings are in `crossing_witness_v1.json`, with the shared adapter outputs
under `fixtures/hydrology/`. These have been delivered directly to Q4/Q5/Q7.

Revision 07 has **96 checked variants with runner repeat equality**, plus the
single final classification inspection. `EVIDENCE.json` enumerates exact runs,
inputs, images, hashes and costs. The complete preceding topology matrix is
revision 06. A full-filesystem incident interrupted the remaining revision-07
rows; the coordinator requested a heavy-matrix pause. Those rows remain pending,
not counted as passes. Q3 reclaimed 756,123,259 bytes of its own redundant
intermediates. `ARCHIVED_OUTPUTS.json` maps removed older BMPs to preserved
unresized RGB PNG witnesses and records the display-alpha conversion. Latest
originals, selected before/after images, fixtures, reports and source assets
were retained. Contact sheets are reproducible with `review.py`.

## Outstanding acceptance and exact ownership

- **Q0/Q6:** per-draw shader composition and shared linear/color/alpha lighting
  boundary; full water Fresnel/specular/receiver integration is still pending.
- **Q2/Q4:** replace diagnostic ground/relief receivers with their source-backed
  surfaces and actual cliff/rock bodies; verify mixed-corner joins and land-family
  transitions without changing Q3 river/shore topology.
- **Q4/Q5/Q7/Q8:** compose source vegetation/buildings with the published exact
  corridor envelopes, retain readable bridge approaches/gates, and validate
  actual transformed footprints/crowns in a plausible gameplay view.
- **Q1:** adopt the selected sampling policy and recheck subtle bed detail and
  reduced-zoom river readability under scrolling.
- **Storage coordination:** complete remaining final-revision matrix rows when
  the shared heavy-run pause is lifted; do not rerun already passed rows blindly.
- **Coordinator promotion:** full source-composed 192-tile/four-phase/two-zoom
  suite, D3D11 parity and full project verification remain unrun. No Integration
  gate is changed and no Civ III patch symbol is required by this Lab candidate.

`SOURCE_PROVENANCE.json` pins assets, code and consumed contracts and separates
source adaptation from diagnostic geometry. `CANDIDATE.json` references the
immutable L21 handoff but does not replace it. No original-art fallback was
selected. Touched Q3 source, fixture, status and audit text was scanned for
personal paths and sensitive information before delivery.

## Follow-up composed defect from Q2

Q2's rejected pale rectangular wet-terrain patch was directly inspected after
the isolated candidate work. Q3 traced it to `max(result, basin)` where the
isolated-water basin is **zero outside support**, clamping negative land coverage
to zero over the loop's tile-aligned candidate range. That also zeroes the coast
relief envelope and creates a lip. This is an exact frozen shore-mask defect,
not Q2 material blending. `FROZEN_BASIN_BUG.md` records coordinates and the
regression; the portable tests prove Q3 leaves the neighboring land classified
as land with continuous height. `systems/hydrology/scene_adapter.h` provides the
explicit sign/unit/lattice conversion for a Q0 opt-in shore callback. Q0 implemented the opt-in callback and Q2 consumed it through its owned wrapper.
The exact matched composed before/after was directly inspected and fixes the
reported defect; see `COMPOSED_REGRESSION.json`. No frozen/v1 file was patched
by Q3. Callback tests additionally prove sign/lattice conversion matches the
full analytic field at 100 samples, including the isolated-water regression.

## Large overview and renewed canonical comparison

User requested the larger 120-tile island overview and then correctly judged
its visual fidelity below the canonical pictures. `island-shores-09` was rendered
at 2048x1280 with MSAA4 at both zooms and directly inspected. It is synthetic
shore topology with diagnostic relief and no river layer, not source terrain.
It exposes too-soft, nearly uniform sand transitions, muddy flat offshore color,
missing authored underwater forms and absent source cliff bodies. The earlier
property review was insufficient for a convincing whole scene; candidate remains
unaccepted. `sea_and_shore.png` and `river.png` were reopened and directly inspected
again against the overview. Canonical water shows readable submerged stone forms,
a dark offshore gradient and a distinct wet/dry shoreline; these static properties
can be restored without inventing or baking a white foam band. Actual rock faces
remain Q4 ownership. Shared shadow silhouettes must come from actual transformed
source geometry/alpha via Q6; no proxy shadow workaround is introduced.

The new independent `context-linear-08` moves radiance into the genuine shared
Q6 MRT output (wire3): no per-material gamma or tone mapping, exposure once after
reconstruction. Directly inspected matched noon average RGB byte difference from
the old diagnostic is .2402 (max9, 3.83% channel values changed), mostly edges.
The separate source material path consumes exact source clip/depth vertices and
Q3 shore fields; its authored bed-detail and depth contrast correction is in
progress. Full source composition will supersede proxy appearance evidence.
