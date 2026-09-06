# Q4 convergence requests

Q4 continues source-backed geometry experiments through `cpp_packet` with
`packet_v1.h` (wire v2). All temporary surface/water inputs are explicitly
synthetic proxies. No upstream implementation is edited.

- **Q0/Q2/Q3:** publish callable continuous surface and shoreline queries or a
  sampled sidecar carrying world coordinates, normals, shore signed distance,
  water elevation, rocky/sandy edge class, stable physical edge ID, wrap period,
  and halo extent. Witness: `fixtures/relief/coast-r1/fixture.json`. Its raised
  coast cannot yet be aligned against a published Q3 rock-foot seam.
- **Q0:** the runner rejects modules with different shader hashes before
  composition (`runner.fixture`: mixed shader contracts require a versioned
  per-draw shader interface). Q4 uses source-body UVs plus world position and
  normals; support per-draw shader identity and a shared world/depth projection
  contract so Q4 can compose with actual Q2/Q3/Q6 modules rather than reimplement
  their systems. Existing `interfaces_v1.h` declarations are proposals only.
- **Q0:** publish verified `test.biq` dataset and region registry with hill-water,
  mountain-chain and biome edges plus a held-out neighbor. At first inspection
  `shared/real_map/` was absent. Synthetic provenance is not acceptance evidence.
- **Q6/Q0:** Q4 consumes `response_v1.hlsl` and the shared environment evaluator,
  but the current BGRA8 target requires display encoding within a draw. Final
  scene-linear composition, common caster/receiver surface, and final response
  belong to Q6/Q0. Heightfield-only shadow witnesses are provisional and do not
  claim exact foliage silhouette shadows.

The initial prompt validation failed on another owner's `in_progress` state;
retry succeeded without a cross-owner edit. This issue is resolved.

## Concrete Q3 real-map classification defect

The newly published real registry has been consumed; its absence is resolved.
`systems/hydrology/field.h:occupancy(hill=true)` currently tests `real==4`, but
both the registry's `NAMES` and frozen source confirm hills are **5**, mountains
**6**, forest **7**, jungle **8** (4 is floodplain). On the verified relief CSV,
Q3 publishes zero rocky coast where the two true hill tiles should contribute.
Q4's sampler consumes this result unchanged; it cannot certify rocky-hill
acceptance on the real region until Q3 corrects the classification. Q4 corrected
its own initial semantic mismatch and retains the rejected real-primary-05 image.
Reproduce with `sample_hydrology.cpp` on `real-primary/terrain.csv`, or inspect
its `.coast` output and compare against the source tile `real` values.

## Current resolution and concrete remaining blocker

- Q3 has corrected hills to 5 and raw-origin/wrap handling. Current source
  inspection and `tests/relief/q3_crop_witness.cpp` pass: the same raw-coordinate
  noise differs by zero between primary and neighboring recaptures. Do not
  report the earlier classification defect as still open.
- Q4 now consumes `Field::sample`, `coast`, and `exclusions` read-only. The owned
  sampler exports Q3's exact capsule clearance radius; the holdout rejects
  source crown extents that cross the published bank envelope.
- Q4 consumes Q5 `source-clearance/clearance.json` and calls Q5's own
  `footprint_intersects` after converting the Q4 lattice footprint to Q5's
  published pixel plane. Q5's current witness has `halo_complete=false`; it
  cannot certify final cropped/wrapped source-tree placement.
- Q2's material-only `scene_adapter.h` hooks are now available and have been
  inspected. They target the frozen provider; there is still no published
  Q4 source-body/height/placement hook that replaces frozen relief atomically.
  Do not layer Q4's diagnostic base plane over Q2's base plane.
- The genuine composition blocker was rerun after those updates:

  `python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/relief/coast-source-r8/q3-composition-blocker.fixture.json --candidate composition-blocker-final`

  Result: `mixed shader contracts require a versioned per-draw shader interface`.
  This fixture selects the real Q4 and Q3 module manifests. Both isolated
  modules are runnable. A Q4-owned fix cannot safely alter this check or the
  shared render graph. Q0 must expose per-draw shaders with one shared projection,
  scene-linear composition/display boundary and geometry/caster ownership, or
  equivalent Q4 geometry/instance hooks into the single frozen scene provider.
  Q4 then supplies source bodies/shoulders/placement while consuming Q2/Q3/Q5
  and Q6 without a duplicate base plane or per-module tone mapping.

No Civ III hook, patch symbol, CSV entry, injected-code change or user screenshot
is required. A coordinator notification was rejected by automatic approval
review because the destination task was considered unverified; these owned
files are the local handoff. No alternate messaging route was used.

## Preserve the composed wet-rectangle regression

Renderer/terrain_lab/v2/audits/terrain/out/Q1/Q2-terrain/composed-complete-wet-r08/h12-z1-pan00.png
SHA-256: ad74e1974fa7b2ecd5e709ff22c93fca906c093da0ac957771a395ea49d1406b

Direct inspection confirms the pale straight-edged lower-right surface and
raised lip. Q2/Q3 subsequently traced the cause to the frozen isolated-water
basin clamping inland signed shore distance to zero outside basin support,
affecting both beach material and coastal relief. This is upstream diagnostic
evidence, not proof of a Q2 weight failure or a source hill-mesh defect. Preserve
this witness and rerun its source-height shoulders after the Q0/Q3 field adapter
lands; do not patch the frozen provider from Q4.

## Continuous derivative adapter for Q0 / Q2

Published `systems/relief/continuous_normal.h`: a generic central world-space
stencil with no local tile clamp, plus the explicit world-Y to frozen-local-v
normal conversion. Q0 should call it at `(tile.column+u,tile.row+1-v)` with step
.006 and the existing horizontal height unit divisor `2*half_width`, sampling
one final continuous ground+relief+hydrology evaluator across the halo. Preserve
flat-underlay semantics explicitly. Both incident tiles must sample the same
field; changing only the denominator or retaining local clamps is insufficient.
Q4's standalone receiver already computes normals on one shared grid.

Read-only reproduction: `audits/terrain/wet_surface_seams.json` reports 408
pairs, maximum normal component delta .421376169 and 145 failures. Dry reports
.826748308 and 153 failures, while heights and shore distances agree exactly.
Q4 supplied the header and a portable regression; it has not edited or connected
Q0's frozen provider. Actual Q2 wet/dry queries and visuals must be rerun by the
owner after adoption. This adapter is not a claim that current shared seams pass.

## City source-vegetation clearance

`CITY_CLEARANCE_RESULT.json` consumes the exact Q7 raw-delta-pixel component
polygons and verifies the BIQ hash/region origin. Holdout r13 removes 12 of 291
source instances, retains 279 with unchanged transforms, introduces zero new
instances and leaves zero retained footprint collisions. The supplied raw city
anchor (20,44) is unchanged. Before/after actual-size images were directly
inspected. Q0 must propagate these exclusions to composed source vegetation;
Q4-only geometry does not prove final Q7 city visibility.

## Selected skin importer correction

`SELECTED_SOURCE_AUDIT.json` proves the normalized skin's two hill LODs were
inherited from Base despite the selected installed skin overriding them. Q4's
`import_selected_relief.py` now extracts the selected field into owned paths.
The shared normalized pack remains read-only. Q0/source-pack owner must adopt
this correction for shared hills. Mountain payloads match, but physical
reconstruction and source material assembly remain unproven (SOURCE_FIDELITY.md).

## Latest interface state: placement consumed, ground replacement pending

Q0 PlacementHooksV1 is adopted in `systems/relief/placement_adapter.h` and
`fixtures/relief/city-clearance-live/fixture.json`. Exact Q7 corridor input is
`audits/objects/metadata/registered-mixed-holdout-v2/corridors.json` (four actual
building polygons, declared clearance 1 pixel, wrap period 6400,0). The owned
compiled cache is hashed through a checked sidecar resource. Actual posed source
vertex hulls are filtered before source geometry and shadows are emitted.
`LIVE_PLACEMENT_RESULT.json` proves four local omissions and no moved/new instances.

Q6 can consume `fixtures/relief/large-source-map-r6/shadow_casters.json` with the
referenced source packet for actual triangle geometry, world/UV offsets, source
alpha and static pose. Placement hulls are expressly not caster shapes.

Q3's latest request is not answered by an approved coast asset scene: actual
source cliff placements exist in coast-source-r15 but fail gaps/repetition and
shoulder joins. Do not turn them into an accepted rock fence or ask Q3 to invent
faces. Q4 must replace its diagnostic receiver with the single Q2/Q3 shared field.
Q0 has offered surface_height/continuous normal and all-tile land coverage hooks;
these need a landed contract and source calibration witness. The earlier mixed
shader rejection is only a legacy-branch reproducer now that Q0 supports declared
linear modules. It must not be cited as proof that the new linear path is absent.
