# Terrain density step

Bounded check complete. Global coarsening demonstrates a useful geometry opportunity,
but changes terrain appearance. The single refinement candidate recovers much of
the sampled appearance and only 4–5 ms of frame time. It is **not promoted**. Accepted
C7 runtime sources and the staged game tuple remain unchanged. The prototype lives
under `Renderer/tools/` and is inserted only into the isolated source snapshot.

## Actual meshes and cost

The normal relief/edge lattice has 64 subdivisions. Ordinary flat interiors already
use `min(16, detail)` with matching dense boundary samples; detailed hill, rocky-shore
and river-neighborhood ground uses the relief lattice. `PatchDetail` selects 32 for
`C3X_RENDERER_PATCH_PIXELS=3` and 16 for `=5` at native128. Zero selects compatibility.
Detail participates in retained signatures and compile keys.

Separate executed fixtures verify the environment values, relief/edge trace, actual
resident layer bytes and Fresh submitted triangles. Occurrence-owner CSV context14
contains 64/32/16; shared-owner context fields are zero, so those fields alone do not
prove shared meshes. Their layer bytes and submitted work do. All arms retain the
same 2,269 owner rows, including 1,103 shared owners and the same canonical identities.

The controls also reduce **hill decal receiver triangles**: `emit_hill_decals` copies
the selected ground/relief receiver triangles. Authored surface recipe placement,
random seeds and decal families remain enabled. This is not a decal-removal control.

Separate 90-frame counts, averaging frames 3–89:

| Representation | Main ground triangles | Main decals | Main mountains | All passes, noon | All passes, night |
|---|---:|---:|---:|---:|---:|
| Normal | 1,759,076 | 798,316 | 665,341 | 10,029,161 | 10,116,972 |
| 32 | 523,345 | 435,157 | 166,288 | 5,788,340 | 5,876,151 |
| 16 | 188,469 | 339,665 | 41,572 | 4,666,633 | 4,754,443 |
| Refinement | 1,519,739 | 798,316 | 526,662 | 9,268,551 | 9,356,361 |

Target/copy footprints stay 28,734,772 / 3,659,240 pixels and rebuild/reuse counts
stay 2 / 1. These are logical footprints, not shader invocation or GPU-time counts.
Coarse geometry slightly changes bounds-based acceptance: at most two records and
512 upload bytes differ in a counted pass/frame. Across 10,528 pass rows per hour,
32 and 16 change non-triangle fields in 12 and 50 rows respectively. Their mean draw
counts rise by 0.103 and 0.345; no feature is disabled.

Refinement changes only triangle/index counts in those same pass rows. Draw counts,
acceptance, instances, upload bytes, targets, copies, rebuilds and reuses are exact.
Main hill decals are unchanged; reflected decal counts differ by 0.46 triangles per
frame on average. Initial adoption still builds 63 owners and reuses 1,040 in all arms.

## Complete-frame comparison

One frozen C7 DLL and one existing client use actual 2240×1260/60 Hz full-guest
borderless presentation and `Present(1)`. All features and the fixed native128
capture/anchors remain enabled: 1,983 records, 759 RENDER, 752 PREFETCH, 472 topology.
Camera remains 0,0; projection performs two 90-frame 1→1.25→1 cycles. Source time is
explicitly 29533 + frame×33 ms. Primary counters, traces, profiling, completion
probes, timestamp queries and readbacks are off.

At each hour, the initial cohort runs normal/32/16 then 16/32/normal. Each run keeps
all 180 frames; warmed tables use frames 3–179, 354 samples per arm. Every warmed
frame misses 16.67 ms. CPU-observed complete-frame measurements in milliseconds:

| Hour / arm | Mean | p50 | p95 | p99 | Worst | Complete mean, all 360 frames |
|---|---:|---:|---:|---:|---:|---:|
| 12 / normal | 142.528 | 140.776 | 148.069 | 317.051 | 877.023 | 141.854 |
| 12 / 32 | 127.170 | 124.842 | 157.571 | 276.379 | 372.626 | 126.027 |
| 12 / 16 | 114.823 | 114.298 | 119.275 | 327.447 | 356.453 | 114.567 |
| 1 / normal | 153.004 | 151.719 | 173.652 | 309.066 | 459.550 | 153.152 |
| 1 / 32 | 135.842 | 134.992 | 142.766 | 285.250 | 317.703 | 134.697 |
| 1 / 16 | 126.997 | 125.755 | 134.473 | 287.701 | 380.703 | 125.941 |

32 saves 15.358 / 17.163 ms noon/night, 10.78% / 11.22%. 16 saves 27.705 /
26.007 ms, 19.44% / 17.00%. Per-block 16 savings are 29.840 / 25.570 ms noon and
23.490 / 28.523 ms night. Outliers and transitions remain included. This cohort's
baseline differs from the earlier causal cohort; the sessions are not pooled.

The one refinement implementation receives its own reversed control/candidate,
candidate/control cohort at each hour with the same client, clock and 180-frame runs:

| Hour / arm | Mean | p50 | p95 | p99 | Worst | Complete mean, all 360 frames |
|---|---:|---:|---:|---:|---:|---:|
| 12 / control | 141.049 | 140.879 | 148.840 | 310.353 | 431.543 | 140.352 |
| 12 / candidate | 135.824 | 134.603 | 143.529 | 379.543 | 422.678 | 135.126 |
| 1 / control | 150.253 | 149.886 | 161.506 | 317.397 | 468.938 | 150.120 |
| 1 / candidate | 146.295 | 145.261 | 187.504 | 337.734 | 432.952 | 145.119 |

Warmed savings are 5.225 / 3.958 ms, 3.70% / 2.63%; timing tails do not consistently
improve. Total triangles decrease 7.58% / 7.52%. This does not meet the substantial
benefit guide. No additional candidate, tolerance sweep or shader probe follows.

## Refinement and quality

The isolated candidate partitions the original 64 grid into fixed world-space
4/2/1-cell leaves. It retains original vertex payloads, the full 64-cell outer ring,
and all finer neighboring boundary points. Indexed and expanded paths agree;
stitched center fans are watertight. No camera-driven resampling, persistent variants
or new retained owner is introduced. Scratch is bounded by the fixed 65×65 grid;
it is local to compilation and discarded before return. Existing cache charges and
retirement own the resulting buffers.

Coarse leaves must bound interpolation against the original piecewise-affine grid:
0.04 native pixels of height, 0.003 normal components, 0.001 other material/UV/effect
fields, and 0.000002 world XY. Corner-fan tests use half these limits; stitching exact
fine-edge vertices consumes the remaining half. Grid and half-grid intersections
bound the original/corner-fan affine difference. These are geometry/payload bounds,
not a guarantee of equal nonlinear shading, silhouettes or displayed pixels.

Early noon color/D24 comparison at zoom 1.122222 precedes the full timing cohort.
32/16 change 302,156 / 429,429 displayed pixels and 649,370 / 723,624 depth pixels.
Mean RGB channel errors are 0.317 / 0.817 levels. The contact crop exposes altered
relief, mountain normals and coast detail. Global coarsening is not accepted.

Separate 90-frame noon/night endpoint snapshots are retained losslessly. Refinement
mean RGB errors are 0.008405 / 0.003404; 611 / 82 pixels exceed eight channel levels.
D24 differs at 76,903 / 76,902 pixels. Depth maximum differences are 29,438 / 1,172,
including visibility-edge changes. It is not pixel-identical or visually accepted.
The inadequate timing benefit stops this candidate before normal-speed transition,
wrap/pressure and production qualification; those gates are not claimed complete.

Sampled warmed first-view means for refinement control→candidate are 3,517→3,673 ms
noon and 3,425→3,472 ms night; geometry preparation rises 2,642→2,808 / 2,538→2,732 ms.
No cold-start improvement is established. At adoption, logical cache bytes decrease
407,725,211→399,652,624 and unique owned GPU bytes 338,461,196→330,338,438; retired/
selection charge remains 156,392. These are not complete physical VRAM bounds.

The local 0 A.D. source at `0ed48b3a1fb1b4b718a78869fa497185af55e086` confirms
`PATCH_SIZE=16`, a 17×17 base vertex grid and two indexed triangles per cell, plus
texture splats/blends (`source/graphics/Patch.h`, `source/renderer/PatchRData.cpp`).
Its 16 cells are different world units from one subdivided Civ III tile. At native128,
C3X's isometric edge projects about 71.55 pixels: regular 64/32/16 edges are roughly
1.12/2.24/4.47 pixels before zoom. Projected interpolation error and coverage govern
useful density; the upstream patch count does not prescribe a C3X tessellation level.

## Evidence and validation

`Renderer/.cache/redraw-density-step/` retains all 3,600 primary raw frame rows,
per-run/pooled complete and warmed distributions, draw/Present phases, source-clock
inputs, initial preparation, adoption receipts, separate counts, owner CSVs and
lossless color/depth archives. `primary-summary.json`, `refinement-summary.json`,
`counts-summary.json`, `quality-summary.json`, contact sheets and difference images
provide the bounded review. The source snapshots and binary identities are retained.
The first candidate link failure omitted the existing export-definition file; its
failed receipt remains alongside the corrected passing build.
All 56 new image archives passed decompressed-byte/hash verification. The 24 Fresh
color/depth archives remain. Only this step's unused automatic Legacy initial images
and reproducible compiler objects were then discarded, with hashes/reasons preserved
in `generated-cleanup.json`; sources, binaries and all timing logs remain. Existing
protected evidence and asset inputs were reused without deletion or duplication.

Three executable tests pass: refinement interpolation/coverage/edges/cancellation,
existing curved-ground tessellation, and hill-decal receiver attachment. Verification
matches all 47,224 accepted input files (12,238,666,273 bytes), 193 accepted runtime
sources, 195 isolated candidate sources, binaries and the staged tuple. No injection,
installation, reference replacement, seasons work, ownership expansion or deferred
milestone work occurs. Native64 wave admission and populated-unit/live-game tests
remain separate work.

Reproduction entry point: `python3 -m Renderer.tools.measure_redraw_density` with
`prepare`, `run --label ... --detail normal|32|16 --hour ... --frames 180`,
`prepare-candidate`, `build-candidate`, and `run --arm candidate`. `--counts --captures`
selects separate diagnostic fixtures. Existing evidence directories are preserved;
the commands refuse to overwrite runs. Primary timing must remain sequential.

## Independent review and next assignment

Independent review reproduces all 3,600 raw primary/refinement frame rows and
their complete/warmed distributions, source-clock inputs and primary options.
All 556 evidence hashes, 193 production and 195 candidate source identities,
three binary identities and four review-file hashes match. Six full color/D24
comparisons reproduce the report. Owner identities, detail fields, layer bytes
and all 10,528 per-hour pass-row comparisons also match. Three executable tests
pass on rerun; both contact sheets were inspected. The receipt is
`.cache/redraw-density-step/auditor-review.json`.

The result supports stopping this refinement branch. It does not reject every
possible LOD approach: conservative interpolation limits retained most geometry,
and a different representation could decouple visible material detail from mesh
density. That larger change has not been measured. The prior assignment's claim
that lattice controls leave decal work unchanged was too broad: authored decal
recipes stay enabled, but hill decals copy receiver triangulation and therefore
lose triangles under global coarsening. The report correctly identifies this.

**Next assignment: eliminate hidden terrain-underlay shading if its measured cost
justifies the change.** In `sandbox/fresh_pipeline.h`, `draw_scene` draws
`geometry_underlay` before natural terrain, mountains and decals. Its
`PSSandboxUnderlay` entry sets `surface_kind=0.5` and calls the hydrology shader's
full `PSMain`, including base material and lighting work. The main underlay submits
about 383,244 triangles in the counted control. The earlier constant-material
probe changed the replacement natural-ground `PSFeature`, not this underlay.
Source ordering establishes a possible hidden-work problem, not its screen
coverage or duration. Some underlay remains necessary for coast/water composition.

First use a small private constant-color underlay probe, preserving actual
coverage, alpha, depth, geometry and all other passes, against the frozen C7
control. Verify the active entry point and its original alpha/clip contract.
Reversed complete-frame measurements establish whether this is a substantial
opportunity; separate color/depth captures locate visible underlay contributions.
If the opportunity is substantial, implement one conservative coverage/depth
approach that suppresses expensive underlay shading only behind genuinely opaque
replacement surfaces. Keep the original path wherever coverage is uncertain.
Do not remove the entire layer, classify occlusion by tile type alone, treat
partial-alpha coast pixels as opaque, or reuse stale camera-space masks. Depth
equality, blending, viewport/scissor transforms, wrapped occurrences and later
coplanar passes need explicit correctness checks. Include the added coverage
work in full-frame timings and stop if it erases the benefit.

This is a change in rendering responsibilities, not another tessellation or
endpoint-shader sweep. The 0 A.D. reference draws its selected base splats and
transition geometry (`TerrainRenderer.cpp:374–383` at the reviewed commit).
It does not establish that our legacy underlay is redundant everywhere, or that
0 A.D. uses the particular depth strategy proposed here. Retain all settled
quality and the existing 16.67 ms target. Full-size continuous zoom remains
roughly 140–153 ms in this cohort; no production speedup was adopted.
