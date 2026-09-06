# Combined terrain checkpoint — 100 real tiles per region

2026-09-06. Sole lead implementation, local Metal. Work remains in progress;
no beauty approval, milestone closure, or Integration promotion is recorded.
The user deferred cities, units and improvements for this terrain pass.
The former small city checkpoint is preserved in `REVIEW.md`.

Current retained work-in-progress: **candidate-v5**, directly inspected across
all three regions at noon/midnight and both gameplay zooms. It improves the
combined scene but does not convincingly match the canonical reference yet.

- [Inland before/after, unscaled gameplay crop](out/gameplay-100-candidate-v5/review/inland-day-comparison.png)
- [Wilderness before/after, unscaled gameplay crop](out/gameplay-100-candidate-v5/review/wilderness-day-comparison.png)
- [Coastal before/after, unscaled gameplay crop](out/gameplay-100-candidate-v5/review/coastal-day-comparison.png)
- [Night comparison with previous best](out/gameplay-100-candidate-v5/review/inland-night-comparison.png)
- [Full 100-tile inland scene](out/gameplay-100-candidate-v5/inland/h12-z1-pan00.png)

Noon changed-pixel counts against the preserved baseline: coastal 131,364;
inland 193,181; wilderness 207,723. Against the immediately preceding complete
best (`candidate-v2`): 63,265; 16,212; 112,933. These are locations of changes,
not quality scores. The linked images show which changes are improvements.
The v3 shoreline regression and v4 residual sawtooth were rejected/superseded;
v5 removes that sawtooth while preserving the softer biome boundary.

## Fixed comparison

`fixtures/beauty/gameplay-100-v1/BENCHMARKS.json` fixes three 10x10 windows of
the verified cached `test.biq`, the 6-tile halo, projection, output sizes and
unscaled gameplay crop. Coastal and inland are terrain foundations for the
future settlement/developed benchmarks, not claims of populated settlements.
Wilderness was selected before tuning this pass. Its first render exposed a
missing-material defect that the earlier wet 128-tile region did not expose.
It is now a regression witness, not a never-seen holdout for future acceptance.

All 300 source tiles remain unchanged. Empty separate object scenario files
preserve the later composition contract. Both 128x64 and 64x32 Civ III tile
bases are rendered, at 1360x800 and 680x400. No camera or object relocation is
used to claim improvement. The full diamond is orientation evidence; the fixed
640x320 crop at native size is primary gameplay evidence. The source map has
no CITY/UNIT records. Existing source vegetation and rivers remain visible.

## Render / inspect / correct record

1. `terrain-128-r01` exposed vertically stretched mountain colors. `r02` uses
   the actual source tiling materials projected onto three world planes,
   weighted by the geometry normal. At full size, rock detail replaces the
   vertical stripes. The isolated mountain comparison changes 44,469 pixels
   in x304–770/y143–508. Geometry and gameplay anchors did not change.
2. `terrain-128-r03` replaces gray river corridors with transmitted bed color,
   water absorption and narrower damp banks. The river now reads as water at
   gameplay size. Its captured curve and water width are preserved. The bright
   uniform edge and limited bed variation still need work.
3. `gameplay-100-candidate-v1` adds the selected skin's actual standard-hill
   height payload and its confirmed 10/14 source height-scale ratio. The
   normalized pack had retained the base skin's hill field. This changes the
   landform, not merely its shading; physical Civ III height calibration is
   still an adaptation requiring review.
4. Wilderness then exposed dark, angular desert patches. An albedo-only render
   reproduced them, ruling out shadow tuning as the remedy. Nine declared
   desert/relief material slots were zero in combined mode because loading was
   gated on the old dunes-only gallery. `candidate-v2` loads them in complete
   composition and replaces the gallery rectangle with BIQ material coverage.
   The patches disappear in actual gameplay pixels. This is the preserved
   previous best for the final boundary/night experiment.
5. `candidate-v3` softened the desert biome boundary but raised a jagged edge
   next to flat water tiles. **Rejected**; its images and recipe remain intact.
   `candidate-v4` adds the compatible flat water join but retains a small
   triangle-interpolation sawtooth at the optical contour. It is superseded
   by `candidate-v5`, which keeps a narrow flat strip across that contour.
   A shared clock-driven
   ambient/moon adjustment also increases night readability without separate
   terrain/object exposure. The isolated coastal night shader experiment leaves
   both noon BMPs byte-identical. This experiment is a C3X lighting adaptation,
   not recovered Civ VI engine behavior.

The latest inspected selection and exact output identities are recorded in
`GAMEPLAY_TERRAIN_EVIDENCE.json`. Image reports retain shader snapshots, packet,
source asset and postprocess hashes. Earlier renders are not overwritten by a
new candidate revision. Numeric evidence supports the visual review; it does
not decide acceptance.

## Three largest remaining visible gaps

1. **Source-backed landforms and joins.** In the wilderness desert, thin seams
   and the inherited analytic dune construction remain. The latter is a
   `diagnostic_proxy`, not approved source art. Selected source ArtDef dune
   controls are known, but their engine construction is not recovered. Source
   mountains still read as isolated peaks, and the new Q4 source cliffs are
   absent from this combined scene. Recover/compose the source construction
   rather than repeatedly adjusting noise, color or sharpening.
2. **Coast and shallow-bed detail.** The coast still has a broad soft band and
   weak recognizable underwater source detail compared with `sea_and_shore.png`.
   Investigate physical bed/surface placement, channel binding and absorption
   together. No screenshot foam or animated water has been added.
3. **Terrain/vegetation composition and lighting contrast.** Forests still
   form square dense clumps, palms are visually dominant, and the ground has
   less source-detail contrast than `hills.png`, `mountain.png`, `river.png` and
   `daynight.png`. Night is more readable but is not a canonical-quality match.
   Preserve actual mesh/alpha-shaped shadows while improving the shared scene.

## Source classification and boundaries

- Selected packs remain `Civ5EnvironmentSkin` and `Civ5EnvironmentVegetation`;
  the extracted standard-hill payload is pinned by SHA-256 in each candidate.
- Ground, tree, water-bed, rock and relief materials reuse source payloads.
  World-plane mountain material projection, hill scaling, river optics and
  the night response are `source_adaptation` / renderer inference.
- The inherited analytic dune body is `diagnostic_proxy`. Its material-binding
  and boundary fixes do not confer source fidelity or beauty acceptance.
- Terrain, topology and object anchors remain Civ III inputs. Historical v1
  shaders/handoffs, native Integration and injected code remain outside this pass.

## Reproduce and verify

```sh
python3 Renderer/terrain_lab/v2/qa/gameplay_terrain.py --region all --revision candidate-v5 --hours 12 0 --output-root Renderer/terrain_lab/v2/audits/beauty/out/replay-candidate-v5
python3 Renderer/terrain_lab/v2/qa/verify_gameplay_terrain.py
python3 Renderer/terrain_lab/v2/qa/present_gameplay_terrain.py
python3 Renderer/tools/renderer_dev.py lab
```

Choose another fresh output root for subsequent reruns; the helper rejects
overwriting a completed report. The verifier/presentation commands inspect the
preserved checkpoint outputs, not the new replay directory.
The presentation helper requires Pillow; rendering and evidence verification
use the existing Mac toolchain. Baseline replay with current shared code pins a
new render identity; preserved original reports/images remain the comparison
authority until deliberately refreshed. Use a new output directory for a
baseline source refresh. Shader-only experiments use `qa/replay_shader.py` and
are explicitly ineligible for promotion.

Checked this pass: Lab workflow (132 Python checks, 12 Node checks and campaign
validation), 18 platform checks, 6 lighting checks, continuous-normal regression,
and a shared-edge/flat-shore regression using the actual combined height kernel
(the preceding geometry fails its negative control),
actual Metal composition, real-map/source/fixture hashes and material-slot
closure. Final four-phase/scroll/topology/repeat/D3D parity, source-fidelity
closure and human approval remain pending. No `full`/injected run is appropriate
for this standalone iteration; LQ0 is not being closed.
