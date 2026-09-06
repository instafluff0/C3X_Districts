# Coast, hill and supplemental volcano iteration

2026-09-06. Terrain-only combined scene, local Metal, sole lead. The latest
retained work in progress is `coast-pass-rocks-r4`; the preceding `candidate-v5`
and every attempted coastal revision remain available. No visual gate is passed.

## Fixed inputs and evidence

The original coastal/inland/wilderness 10x10 benchmarks remain fixed by
`fixtures/beauty/gameplay-100-v1/BENCHMARKS.json`. All terrain bytes, scenario
placements, camera parameters, output sizes and source packs match their
previous best. The new long coast is raw BIQ origin `[64,38]`, extent 16x8,
halo 6, 128 visible tiles including 55 water tiles and 48 land/water edges.
It was selected before this pass's shoreline tuning. Its 1616x888 / 808x444
outputs and native 640x320 crop are now pinned by
`fixtures/beauty/coast-pass-foundation/BENCHMARKS.json`; it becomes a regression
witness after this pass, not a never-seen holdout for later acceptance.

All four regions have noon/midnight and both gameplay zooms. Original long-coast
noon baseline is `coast-pass-baseline-v5`; its matched night baseline is a fresh
`coast-pass-baseline-v5-night` output. Baseline images are never overwritten.
Reference inspection used canonical `sea_and_shore.png`, `civ3_real_example.jpg`,
`hills.png` and the prior mountain/day-night evidence. The paired canonical
overview is labeled as resized navigation; gameplay comparison crops are unscaled.

## Render / inspect / change

1. `shore-r1`: continuous world-coordinate displacement at three spatial scales
   changes the coast outline and narrows the beach band along the contour.
   Protected land/water centers and native edge connections stay intact. Direct
   long-coast inspection found more variation but an overly soft water rim.
2. `rocks-r2`: the actual six normalized cliff source assets are available via
   an opt-in bundle; four large variants are placed. Each draw binds its own
   base color, LEAN0, LEAN1 and gloss. The source gloss DDS was sRGB-tagged data;
   the loader now permits a compression-compatible linear view, preserving the
   source payload and recording both source and view formats. The first image
   showed tiny, regularly spaced stones: **rejected as a finished cliff result**.
3. `rocks-r3`: larger uniform source bodies, deeper burial, stricter hill
   association and spacing make larger groups. They still look like detached
   boulders on a low bank. This does not solve the underlying terrain join.
4. `rocks-r4`: stronger bounded shoreline displacement makes the long headland
   and fixed coastal inlets visibly more articulated. Direct selected-source
   hill sampling replaces the inherited nonlinear height threshold and local
   crop UVs. Lower source relief now survives, including in the dry wilderness.
   Source mapping uses canonical raw coordinates with an exact wrap period for
   the verified 100-wide map. The physical amplitude and UV scale remain C3X
   adaptations, not recovered engine parameters. Noon/night inspection retained
   this as an incremental combined improvement, with cliff continuity unresolved.
5. `volcano-witness-r1`: source inventory proves zero volcano tiles in test.biq.
   A separate, explicitly synthetic 100-tile inland copy changes only real
   terrain 6 to 10 at local `[5,5]`; no verified benchmark terrain is altered.
   The first render exposed independent height rotation/aspect and material UVs.
6. `volcano-witness-r2`: height and all four appearance channels now use the same
   local-v orientation and uniform `.62` footprint. The source crater opens up
   and the source surface feature moves to its corresponding landform. At noon
   the change is visible around full-image x708–786/y330–398. The side surfaces
   remain too smooth/streaked, the physical source extent remains unproven, and
   no engine-equivalent eruption, emissive light or effect is claimed. This
   diagnostic is kept separate from real-map beauty acceptance.

The retained shoreline changes are visible at full long-coast x565–698/y417–597,
and fixed coastal x432–552/y180–346. Rocky-versus-sandy differentiation is visible
around long-coast x628–956/y520–598 and x938–1068/y655–728. The remaining pond
rock rings are visible in that latter area; the crop deliberately retains them.

Noon changed pixels against previous best: coastal 219,664; inland 174,509;
wilderness 176,296; long coast 321,940. These include water response, relief and
shadow changes and are **not quality scores**. `COAST_rocks-r4_EVIDENCE.json`
pins every output identity and provides matching night/zoom results. Complete
cliff material draws are present in the composed packets; 8 coastal, 8 inland,
0 wilderness and 22 long-coast instances are recorded. Instance counts do not
establish plausible composition.

## Engineering checks and next visible work

Passed: Lab workflow (132 Python, 12 Node and campaign validation), 18 platform
checks, 6 lighting checks, 3 hydrology checks, actual combined source-coordinate
regression and shared-edge/flat-shore regression. Both shoreline profiles preserve
tested tile centers, native connections, crop translation and horizontal wrap.
The new coordinate test exercises the actual CPU volcano mapping and direct hill
block. Actual composed packet metadata verifies all four cliff channels and the
linear gloss view. No full/VM/injected verification or promotion was requested.

Next loop must address the hill/cliff join as geometry and composition, not add
another layer of random stones. The current coastal envelope lowers hill terrain
before it reaches the optical shore; source rocks therefore lack an elevated
grassy shoulder. Preserve the flat shared water join while deriving the shoulder
and exposed face together. Match the reference's rock material response through
source channel interpretation and shared lighting. Also investigate why the
shallow band hides source-bed detail. The analytic dune proxy and physical
mountain/volcano reconstruction remain separate unresolved source-fidelity work.

## Reproduction

```sh
python3 Renderer/terrain_lab/v2/qa/coastal_pass.py --region all --revision rocks-r4 --hours 12 0 --output-root Renderer/terrain_lab/v2/audits/beauty/out/replay-coast-r4
python3 Renderer/terrain_lab/v2/qa/verify_coastal_pass.py
python3 Renderer/terrain_lab/v2/qa/present_coastal_pass.py
python3 Renderer/terrain_lab/v2/qa/volcano_witness.py --revision r2 --prepare-only
python3 Renderer/tools/renderer_dev.py lab
```

Use a fresh output directory; completed render reports are protected. Presentation
requires Pillow. Source-derived cliff bundles are locally ignored and reproducible
from the existing normalized source pack. Shader/source snapshots, provenance,
previous images and rejected experiments remain preserved. No agent delegation,
production changes, future wonders or human approval belong to this pass.
