# Q8 coastal checkpoint r01 — preserved history, not accepted

Current terrain-only combined work is recorded in
[CURRENT_VISUAL.md](CURRENT_VISUAL.md): three fixed 100-tile `test.biq` regions,
matched gameplay-size comparisons, source material corrections and explicit
remaining defects. The historical city/route observations below remain valid
for their original fixtures; the user deferred those layers for the current pass.

## Water recheck r03: no additional visual change

The source-water update notification was tested in the identical r02 fixture as
`out/coastal-r03-water/view.png`. Its displayed BMP SHA-256 is exactly the r02
value (`753b3a65ca0b23a1f05d7fd8d0de3bb2d085c59ecfb8d578d3f258822f09812b`).
Fixture, module, settings, backend, packs, postprocess and shader hashes match;
the platform contract hash changed. The latest Q3 material was already consumed
by r02. No extra water improvement is claimed from this fourth single frame.

## Latest: r02 adopts the new city

`out/coastal-r02-after/view.png` now includes Q7's compatible full-source city
module and Q6's final composed caster/receiver pass, plus Q3's requested raw-origin
material anchoring. This third small Mac frame was rendered and directly inspected.
The same terrain, city anchor, roads, camera, noon phase, zoom and sampling remain.
The old city layer is empty so there is no duplicate body. Exact recipe and source
geometry hashes are in `fixtures/beauty/coastal-r02/RECIPE.json`.

Visible gain: the old tall monolithic city prop is replaced by separate legible
source buildings; shadows extend from actual objects across the terrain. Visible
remaining defect: this town now reads as four disconnected small buildings rather
than a cohesive settlement. Q7 should improve grouping within the same anchor and
route clearance. The soft shore, weak underwater detail, cut-looking hill-water
edge, and road/vegetation readability remain. This is a partial improvement, not
Civ VI-level beauty or Q8 acceptance. Q4/Q5 new candidates are still not composed.

The r01 observations below remain the preserved first checkpoint, not current
claims that Q7 is still incompatible. No heavy matrix was launched: three single
592x376 frames total. No new repeated full suite should displace bounded visual
fixes, especially while the shared host is under transient memory/disk pressure.

The coordinator now owns combined Q8 review. This is a small real-map coastal
checkpoint, not a completed developed-gameplay fixture or a promotion request.
Both images were rendered on Mac and directly inspected at actual output size.

## Visible result

- The current candidate has darker offshore water and clearer geometry-shaped
  terrain/tree shadows than the matched prior source-linear scene.
- This is not yet a convincing Civ VI-quality coast. The shoreline remains
  soft, the raised grass edge reads as a cut rather than a source-rock cliff,
  and recognizable underwater detail is weak.
- The inherited city still reads as a tall isolated prop rather than Q7's new
  settlement assembly. Parts of the road are hidden in the terrain/vegetation.
  These visible failures outweigh small material/sampling refinements.

Before: `out/coastal-r01-before/view.png`.
Current candidate: `out/coastal-r01-after/view.png`.

The pair fixes verified `test.biq` terrain, augmentation, camera, 592x376
viewport, noon, zoom, 4x MSAA, anisotropy8, bias0 and Q1 linear reconstruction.
Only fixture identity and module selection differ. It compares multiple
intentional system changes together, not an isolated-variable experiment.
The before is a prior source-linear configuration, not an immutable v1 capture.
Per-run reports preserve render/cache identities and shader snapshots; the
review verification script checks matched inputs and output hashes.

## Three bounded visual priorities

1. **Source-backed coastal form and detail — Q3/Q4.** At native image pixels
   approximately x240–325/y110–275, compare lowland beach, hill water edge and
   the sharp land-side drop. Replace the diagnostic-looking cut with coherent
   source-rock hill-coast geometry and expose useful authored shallow-bed detail.
   Keep terrain semantics/camera unchanged. Q2's reported normal discontinuities
   remain part of the terrain/relief convergence check, not accepted here.
2. **Actual city assembly — Q7/Q0/Q6.** Around x270–325/y45–135, replace only
   the inherited city body with Q7's compatible source-backed settlement at the
   same anchor. Verify the body contributes to the common caster/receiver pass.
   Declared but unbound material channels remain an explicit source-fidelity gap.
3. **Readable route and vegetation clearance — Q5/Q4.** Around x340–455/y160–265,
   make the connected road visible through its vegetation corridor using actual
   source-tree extents; no draw-order masking, whole-tile clearing or new art.
   Use the same recipe and source terrain so this is not a favorable relocation.

These are bounded fixes against one image, not requests for more large isolated
matrices. Re-render this checkpoint after owner delivery. Preserve wider
topology, phase, zoom, heldout and native parity gates for actual acceptance.

## Explicit limitations

Q2/Q3/Q6 current material, shore and world-shadow paths are composed. Q4 new
source cliffs, Q5 new route presentation, and Q7 new city geometry are NOT yet
in this frame. Their absence is not evidence of failure of an unseen candidate;
it is a missing composed result. Legacy city/road/relief source bodies remain
declared baseline inputs. No new art is generated by Q8's include-only wrappers.

The 4x4 view is intentionally bounded and still has a black map boundary. It is
a first diagnostic gameplay-context scene, not the desired filled developed
viewport, two-settlement fixture, HUD/label-clearance proof or full beauty gate.
No night, reduced-zoom, scrolling, heldout, or repeat pass is claimed for r01.
No Integration files, injected code, global gate, or historical handoff changed.

## Commands

```sh
python3 Renderer/terrain_lab/v2/qa/prepare_checkpoint.py coastal-r02
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/beauty/coastal-r01/before.fixture.json --candidate coastal-r01-before --output Renderer/terrain_lab/v2/audits/beauty/out/coastal-r01-before
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/beauty/coastal-r01/after.fixture.json --candidate coastal-r01-after --output Renderer/terrain_lab/v2/audits/beauty/out/coastal-r01-after
python3 Renderer/terrain_lab/v2/qa/verify_checkpoint.py
```

City adoption was prepared with `python3 Renderer/terrain_lab/v2/qa/adopt_city.py`
and rendered with the same quick command using
`fixtures/beauty/coastal-r02/after.fixture.json` and its own r02 output namespace.

Use a new version for new inputs; retain the r01 reports/images. The initial
restricted process could not access Metal; GPU-enabled Mac execution succeeded.
No VM was used. Future owner changes may require refreshing versioned wrappers;
matching file names alone never establishes a controlled comparison.
