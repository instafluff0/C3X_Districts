# Q4 source-art candidate audit — not accepted

Q4 remains blocked at cross-system convergence, with source-backed experiments
and explicit unresolved visual items. This is not a completed visual track or
an Integration handoff. No global milestone, v1 handoff or accepted audit changed.

## Contextual evidence first

All paths below are relative to `Renderer/terrain_lab/v2/audits/relief/`.

- Primary actual terrain: `out/Q1/Q4-relief/real-primary-09-check/`.
- Fixed neighboring holdout: `out/Q1/Q4-relief/real-holdout-10-check/`.
- Latest source forest scale correction:
  `out/Q1/Q4-relief/real-holdout-12/context.png` (actual-size crop).
- Source-rock coast, latest source-piece placement refinement:
  `out/Q1/Q4-relief/source-coast-11/context.png` (actual-size crop).
- Matched real-terrain before/after source-material correction:
  `real-primary-06` versus `real-primary-09-check` under the same output root.
  Camera, viewport and terrain are unchanged. Material selection, source footprint
  masking, source-instance placement and receiver fixes are the declared variables.
  The earlier image is a rejected diagnostic, not an accepted reference.

The actual BIQ is `test.biq`, SHA-256
`a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e`,
17,173 bytes, 100x100, horizontal wrap. Q0 registry regions are `relief` at raw
(14,42), 4x4 plus two-tile halo, and `relief-holdout` at (18,46). The latter was
selected before tuning and is now a regression witness, not an untouched holdout.
Source terrain/river records are unmodified. No cities, roads, units or improvements
were added to those BIQ records. Trees and relief are renderer interpretations of
existing terrain identities. Raw-coordinate tree seeds are stable across crops.

These are provisional terrain contexts, not developed-gameplay beauty proof.
Q8's shared developed recipe was unavailable in the owned fixture inventory at
inspection; Q4 has not invented a city/network composition to replace it.

## Render / inspect / correct sequence

1. Initial diagnostic opacity failed: unused terrain texture alpha was treated
   as cutout coverage. The black holes were inspected and fixed.
2. Generated exposed cliff faces looked like a repeated textured wall. They
   were rejected, then explicitly disallowed by the user. They are not selected.
3. Replaced all visible coast rocks with actual normalized Civ VI cliff meshes.
   Source UVs and uniform transforms are tested. Early procedural hills were
   replaced by the extracted standard source height field.
4. First real-map receiver had a terrain-ID mismatch; Q4's mapping was corrected.
   Q3 independently corrected its hill classification/origin handling. Current
   Q3 recapture witness passes; neither problem remains listed as a blocker.
5. Mountain-top material applied over entire bodies caused pale horizontal
   striping. Source base material, source summit thresholds and source footprint
   coverage removed that defect. Body positions and source UVs are unchanged.
6. Actual source rocks were composed with subordinate grassy joins. A later
   uniform scale plus deeper burial exposes broader source faces without
   deforming a body or generating substitute rock geometry.
7. Dense holdout exceeded the packet's 256-buffer cap. Material batching fixed
   it while preserving source instances. Current packets use Q0's verified
   content-reference format; duplicate payloads are hard-linked locally.
8. Q3 river/bank capsules and Q5 full route/bridge envelope tests now reject
   transformed crown extents. The source-only Q5 witness rejected 51 of 95
   candidate trees, retaining 44, before the later forest scale refinement.
   The initial river holdout rejected 67 instances. No origin-only test or
   draw-order trick is used. The latest forest scale increases the source
   broadleaf bodies uniformly; its new full matrix remains pending.

## Direct visual findings

- Source hill heights give broad, low connected rises without per-tile raised
  diamonds. The finite diagnostic viewport boundary is still visible in overview
  images; contextual crops are the primary inspection view.
- Source cliff meshes replace the invented wall. Their real irregular faces,
  UVs, and feet are visible. Some runs still read as separate masses; seamless
  hilltop/rock/actual-water composition is not signed off on the proxy receiver.
- Source mountain heights retain steep spires/ridges. The material-striping and
  rectangular source-background defects were corrected. Separate source-body
  feet/terrain material and normal continuity still need the shared Q2/Q4 scene.
- The latest forest scale creates a connected broadleaf canopy with distinct
  palm/understory silhouettes. Earlier small forest instances looked sparse and
  were rejected. Exact foliage cast shadows and final cutout/normal response
  remain shared Q6/platform work, not a passed beauty criterion.
- Noon, sunset, midnight and sunrise remain distinct and readable. The current
  source material color stays intact; night is cooler and darker.
- Direct source-height dune diagnostics show fine source detail but do **not**
  supply the required broad directional crests. The source volcano witness does
  **not** prove original physical proportions. Marsh is not accepted. These are
  explicit open source/recipe items; no invented fallback has been selected.
- The four coast orientations, cove and island diagnostic images were directly
  inspected. Back-facing rock runs are naturally occluded. Wrap/hill acceptance
  and the final mixed rocky/sandy joins remain pending unified real-map composition.

## Executable evidence

`METRICS.json` contains 32 phase/zoom records from four focused checkpoints:
source coast, real primary, real holdout and source vegetation/route clearance.
Each `check` ran 24 frames (four phases, both zooms, origin/pan/return) plus
backend repeats. Every backend repeat passed; every return-to-origin hash is
identical. Maximum aligned-pan mean channel error is 0.000074 byte, with sparse
raster-edge maximum 21/255. These metrics do not constitute visual acceptance.

Commands from the repository root:

```sh
python3 Renderer/tools/lab_v2.py prompt Q4-relief
python3 Renderer/tools/renderer_dev.py state
# C3X_LAB_PYTHON identifies a local Python with numpy and Pillow.
"${C3X_LAB_PYTHON:-python3}" -m unittest discover -s Renderer/terrain_lab/v2/tests/relief -v
"${C3X_LAB_PYTHON:-python3}" Renderer/terrain_lab/v2/systems/relief/build_fixture.py coast-source --revision 11
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/relief/coast-source-r11/fixture.json --candidate source-coast-11
python3 Renderer/terrain_lab/v2/app/runner.py check --fixture Renderer/terrain_lab/v2/fixtures/relief/real-primary-r9/fixture.json --candidate real-primary-09-check
python3 Renderer/terrain_lab/v2/systems/relief/run_checks.py
"${C3X_LAB_PYTHON:-python3}" Renderer/terrain_lab/v2/systems/relief/evidence.py
```

All fourteen focused tests pass (six portable clearance tests, two C++ adapter
regressions, and six local source integration checks). Source integration tests skip explicitly if local packs are
unavailable; no Firaxis payload is redistributed with tests. The latest source
forest and cliff placement refinements have native-noon evidence; their repeated
full phase/zoom matrix is **pending**, not inherited from older versions.

Do not launch `run_checks.py` during the shared storage pause without checking
space and coordinating heavy output. It is a focused Mac checkpoint, not the
192-tile promotion suite. The latter and Windows parity were not run because
composition/visual acceptance is blocked; no closure or promotion is claimed.
Injected C did not change and was not compiled.

## Stop condition and handoff

`INTERFACE_REQUESTS.md` records the exact, rerun cross-owner blocker: the real
Q4/Q3 module composition fails before rendering with `mixed shader contracts
require a versioned per-draw shader interface`. Q2's new material hooks do not
provide a Q4 source-body/height/instance replacement hook. Q4 cannot fix the shared
render graph or paint another base plane over Q2 within its ownership.
Q5's current published corridor witness also declares `halo_complete=false`.
Final cropped/wrapped vegetation acceptance cannot be asserted from that witness.

`Q4_SOURCE_CANDIDATE.json` is a proposed **provisional** handoff for those owner
interfaces, not an accepted replacement. It references immutable L21 only.
Remaining local visual and source-recipe work is enumerated above; it must resume
in the composed source-backed scene rather than treating proxy beauty as closure.

## Storage and hygiene

Q4 used Q0's read-only `compact_packet` API on Q4-owned fixture packets only.
Thirty-five packets externalized 2,410,906,596 logical bytes into a Q4-local
content store, reducing the fixture tree from about 2.5 GiB to 923 MiB. The
source payloads remain recoverable. Reports and original candidate images were
preserved. Forty-eight redundant repeat-image copies (94,374,432 bytes) were
removed after their successful deterministic checks; primary images and recorded
hashes remain. Q0 shared content, other owners' outputs, and source art were not
removed. The final shared free-space check exceeded 8 GiB, reflecting concurrent
owner cleanup as well as Q4's recovery; it is not attributed entirely to Q4.

Source-derived packets, content blobs and render images are locally ignored.
Checked-in candidate text uses repository-relative paths and contains no personal
home paths, usernames, credentials or machine identifiers. Generated backend job
files remain ignored local runtime metadata. See the final hygiene scan result
in `Q4_SOURCE_CANDIDATE.json`.

## Latest source-fidelity correction (revisions 13–15)

The user's concern about the canonical mismatch led to a direct installed-package
comparison. `SELECTED_SOURCE_AUDIT.json` finds both hill LODs in the normalized
selected-skin pack are wrong: they contain Base data instead of the selected
skin override. The owned importer now extracts the actual selected field.
Directly inspected `hill-base-versus-selected.png` demonstrates the difference.
The newer field and its 10/14 relative source amplitude were rendered at native
noon in `source-coast-14`; this exposed sand beneath designated hill edges.
Revision 15 makes the diagnostic edge classification topology-based and places
source rocks at the lower source hill heights. `source-coast-15/context.png`
was directly inspected: the unwanted middle sand ribbon is removed, but gaps,
repeated source rock clusters and a straight grassy face remain. It fails beauty.
No broad matrix was run for these rejected changes.

Source mountain HM/HBLEND/ID files match all 30 installed selected-skin channels,
but that does not establish correct physical reconstruction or material assembly.
The current mountain body is visibly too narrow/spiky compared with canonical
`mountain.png`. Its hard-coded aspect interpretation is now explicitly diagnostic.
The typed source entry's height_scale 25 and ArtDef height/width 32/42.399979
require an identified composition relationship. Do not solve this by deforming
invented replacement peaks or claim exact source models from heightfield hashes.

Before that source correction, city witness r13 removed exactly 12 overlapping
source trees; all 279 survivors retained their transforms. The r12/r13 contextual
images were directly inspected and confirm a local opening, with surrounding
canopy preserved. Actual Q7 city composition is pending, not demonstrated by an
empty opening. `CITY_CLEARANCE_RESULT.json` carries the exact provenance.

The Q0/Q3 composition blocker was rerun and still fails before rendering with
`mixed shader contracts require a versioned per-draw shader interface`.
A locally published Q0/Q2 interface request includes a continuous derivative adapter and the
exact incident-edge seam evidence. No cross-owner source was modified.

Additional exact commands:

```sh
python3 Renderer/terrain_lab/v2/systems/relief/import_selected_relief.py
"${C3X_LAB_PYTHON:-python3}" Renderer/terrain_lab/v2/systems/relief/build_fixture.py coast-source --revision 15
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/relief/coast-source-r15/fixture.json --candidate source-coast-15
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/relief/real-holdout-r13/fixture.json --candidate real-holdout-13
```

## Requested large map and continued work

The requested demonstration is a **synthetic 11x11 layout (121 tiles)**, rendered
at 1920x1280 and the second zoom. It is not captured BIQ state. The runner caps
ordinary composed previews at 128 tiles; initial 256/169-tile preparations were
not valid ordinary-render candidates and were not promoted or misreported.

Latest image: `out/Q1/Q4-relief/large-source-map-06/map.png`.
Iteration overview: `large-map-iterations.png`. All eight r3/r4/r5/r6 phase/zoom
images were directly inspected, using contextual overview and native detail.

- r3 preserves the full-extent interpretation of source MountainWidth. Its
  narrow spires and stretched planar rock mapping fail the canonical benchmark.
- r4 uses the existing source rock base/top textures with world triplanar
  projection and source material-height normal detail. Body height data and
  instance positions were unchanged; surface detail improved. The material
  projection remains an inferred C3X adaptation, not a recovered engine shader.
- r5 tests **MountainWidth as half-extent**, preserving all HM values and peak
  heights. Broader overlapping bodies produce connected ridges substantially
  closer to the canonical silhouette. This is an explicit, unconfirmed source
  coordinate hypothesis; it is not an accepted source-body normalization.
  Forest positions that newly collide with the broader bodies are omitted.
- r6 derives depth from the actual projection's view ray, replacing unrelated
  hard-coded XY and Z depth slopes. It retains r5's hypothesis and source data.
  Small shoulder artifacts remain and source silhouette shadows are still pending.

Exact commands:

```sh
"${C3X_LAB_PYTHON:-python3}" Renderer/terrain_lab/v2/systems/relief/large_map.py --revision 6 --source-extent radius
python3 Renderer/terrain_lab/v2/app/runner.py compose --fixture Renderer/terrain_lab/v2/fixtures/relief/large-source-map-r6/fixture.json --candidate large-source-map-06
```

`fixtures/relief/large-source-map-r6/shadow_casters.json` publishes each actual
triangle-list buffer, transformed world-position/UV offsets, current static pose,
source alpha bindings, and source footprint clipping for Q6. It explicitly rejects
placement hulls as shadow geometry. The existing height-atlas shadow stays
provisional; no missing silhouette shadow is called accepted.

## Live shared placement adoption

Q0's newly published PlacementHooksV1 is now consumed by
`systems/relief/placement_adapter.h`. It receives every transformed source vertex,
constructs the XY convex hull, and tests provider polygons with declared clearance
and wrap aliases. `prepare_placement.py` pins provider JSON sidecars and a hashed
compiled cache. Exact copied terrain, city and scenario bytes preserve Q7 state.

Controlled renders `city-clearance-off-01` and `city-clearance-live-01` use the
same current shared provider, viewport, source terrain and city anchors. The
callback removes four of 597 instances, leaving 593 with unchanged source
transforms. It changes 478 pixels. Both current images already show the town;
the earlier Q7 screenshot is not a matched baseline, and its visibility change
cannot be attributed solely to Q4. `LIVE_PLACEMENT_RESULT.json` records this
limited, successful locality result. It is not complete route/river/phase/wrap
or city-readability acceptance.

```sh
python3 Renderer/terrain_lab/v2/systems/relief/prepare_placement.py --fixture Renderer/terrain_lab/v2/fixtures/objects/real-holdout/fixture.json --name city-clearance-live
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/relief/city-clearance-live/fixture.json --candidate city-clearance-live-01
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/relief/city-clearance-live/off.fixture.json --candidate city-clearance-off-01
```

Shared interfaces changed during this work. Q0 now admits mixed modules declaring
its linear color contract; the earlier legacy mixed-shader error is not proof
that all per-draw composition remains unavailable. Q4 still needs migration to
that projection/color boundary and a single shared ground/relief receiver. The
published scene hooks currently lack continuous surface height/normal callbacks;
Q0 has offered that interface but it has not been consumed or proven here.
Q3's reproduced land-coverage notch cannot be closed by layering a second Q4
ground plane. No final relief or coastal beauty acceptance is claimed.
