# Redraw causal isolation

Completed bounded standalone diagnostic, independently reviewed; production renderer sources and the staged game tuple were not changed. The control is accepted C7. The result supports a large raster/material opportunity, **not a demonstrated image-preserving 108–118 ms optimization**.

## Experimental contract

One common client uses the actual 2240×1260, 60 Hz full-guest borderless display and Present(1). Native128 capture retains 1,983 records (759 RENDER, 752 PREFETCH, 472 topology-only), fixed anchors/camera, authoritative projection, and the explicit 29533 + frame×33 ms clock. Each primary run contains two 90-frame 1→1.25→1 zoom cycles. Noon and night each use A/B/C/D followed by D/C/B/A, sequentially. Counters, traces, profiling, completion probes and timestamps are off. Readback occurs in separate fixtures after timing.

- **A:** private normal C7. Frozen-C7 sanity runs in both orders expose no added slowdown: warmed means frozen/private 135.642/134.771 ms and private/frozen 134.452/134.749 ms.
- **B:** empty late raster scissor at geometry submission, preserving original CPU bounds, selections, shaders, bindings, updates, uploads, draws, vertices and indices.
- **C:** the same scissor setup, suppressing only those geometry draws. Original CPU work, binding/upload calls and intended-work counters remain.
- **D:** existing whole-reflection omission, covering reflected material, scene and units on this corrected capture.

Ten draw sites in `issue_records`, vegetation and unit-body paths are transformed only in private source snapshots. Shadow preparation, fullscreen relighting/reconstruction, clears, copies and publication stay outside the geometry scopes. `main_material` is not exercised by this C7 fixture; the reflected material pass is exercised. No runtime ownership or live-game qualification is expanded.

## Complete-frame timing

Each warmed row combines 354 frames (177 from each block). No outliers are removed. Quantiles use the retained harness’s lower-order statistic convention. Deadline counts use exactly 1000/60 ms; 16.67 is the rounded display label.

| Hour | Arm | Mean ms | p50 | p95 | p99 | Max | Misses >16.67 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 12 | A | 134.346 | 133.281 | 141.588 | 147.786 | 466.860 | 354/354 |
| 12 | B | 26.269 | 27.652 | 33.955 | 48.428 | 52.209 | 351/354 |
| 12 | C | 23.929 | 16.760 | 33.662 | 50.859 | 95.054 | 213/354 |
| 12 | D | 121.504 | 121.890 | 130.300 | 134.981 | 268.751 | 353/354 |
| 1 | A | 145.850 | 144.387 | 160.193 | 271.543 | 520.657 | 354/354 |
| 1 | B | 28.189 | 28.584 | 34.132 | 49.776 | 127.490 | 354/354 |
| 1 | C | 23.470 | 16.742 | 33.667 | 35.082 | 52.106 | 203/354 |
| 1 | D | 129.582 | 129.693 | 147.726 | 156.397 | 271.471 | 353/354 |

All 360 frames per row, including the first-three transitions, remain in `all-primary-frames.csv` and `primary-summary.json`; the JSON also retains every block’s full/warmed mean, p50/p95/p99/max/miss counts and CPU draw/Present distributions. Complete distributions are:

| Hour | Arm | Mean ms | p50 | p95 | p99 | Max | Misses |
|---|---|---:|---:|---:|---:|---:|---:|
| 12 | A | 133.178 | 133.276 | 141.593 | 148.292 | 466.860 | 358/360 |
| 12 | B | 26.966 | 27.705 | 34.038 | 48.916 | 172.754 | 355/360 |
| 12 | C | 24.186 | 16.756 | 33.692 | 51.319 | 133.858 | 215/360 |
| 12 | D | 120.879 | 121.727 | 130.317 | 141.872 | 273.236 | 358/360 |
| 1 | A | 144.716 | 144.067 | 160.193 | 271.543 | 520.657 | 358/360 |
| 1 | B | 28.767 | 28.584 | 34.275 | 85.680 | 165.276 | 358/360 |
| 1 | C | 23.235 | 16.741 | 33.667 | 35.082 | 52.106 | 205/360 |
| 1 | D | 129.162 | 129.667 | 148.513 | 254.355 | 272.088 | 357/360 |

A→B gains 108.077 ms noon and 117.660 ms night. B→C adds 2.339/4.719 ms; A→D gains 12.842/16.268 ms. Reversed block results preserve that ordering. CPU draw/Present means are A 21.543/112.803, B 18.838/7.430, C 8.373/15.557, D 18.391/103.114 ms noon; A 22.989/122.861, B 21.394/6.796, C 8.317/15.153, D 19.636/109.945 ms night.

**These deltas are not additive GPU pass timings.** Empty scissor can allow driver work elimination before fragment shading, and changed geometry outputs affect downstream work. CPU intervals contain queue/backpressure shifts; Present(1) quantizes the near-budget B/C results. C retains CPU loops, state changes/uploads, shadow preparation and fullscreen/copy/publication work, so its residual cannot be assigned solely to fullscreen GPU work. No timestamp or completion-query attribution is claimed.

## Work and output invariants

Separate 90-frame fixtures prove frozen C7=A=B=C for every intended-work row: 10,528 rows per hour, including selection, submitted instances/indices/triangles, tracked upload bytes, clears, target/copy footprints and rebuild/reuse counters. Original binding/update/Map/Unmap/clear/copy calls are identical frame by frame. B adds only diagnostic RS operations; C changes only scoped geometry Draw calls relative to B. All 39,600 B/C scissor scopes in the separate instrumented runs verify enabled scissors, exact restoration and zero leaked scope depth; no original RS setters run inside them. Those verification getters and checks are absent from primary timings.

| Hour / arm | Actual draws incl. publication | Intended triangles | Original bind/state calls in fresh draw | UpdateSubresource calls in fresh draw | Tracked upload bytes | Clear/target pixels | Copy pixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| 12/A | 5,159.84 | 10,029,161 | 31,268.13 | 1,464.10 | 2,172,074 | 28,734,772 | 3,659,240 |
| 12/B | 5,159.84 | 10,029,161 | 31,268.13 | 1,464.10 | 2,172,074 | 28,734,772 | 3,659,240 |
| 12/C | 6.00 | 10,029,161 | 31,268.13 | 1,464.10 | 2,172,074 | 28,734,772 | 3,659,240 |
| 12/D | 3,410.26 | 5,475,042 | 18,737.30 | 989.69 | 1,299,949 | 25,904,056 | 2,850,464 |
| 1/A | 5,527.31 | 10,116,972 | 32,374.40 | 1,464.10 | 2,172,197 | 28,734,772 | 3,659,240 |
| 1/B | 5,527.31 | 10,116,972 | 32,374.40 | 1,464.10 | 2,172,197 | 28,734,772 | 3,659,240 |
| 1/C | 6.00 | 10,116,972 | 32,374.40 | 1,464.10 | 2,172,197 | 28,734,772 | 3,659,240 |
| 1/D | 3,592.22 | 5,518,659 | 19,287.02 | 989.69 | 1,300,073 | 25,904,056 | 2,850,464 |

The API trace covers the fresh draw. Publication executes afterward: its logical counter proves one draw, a 32-byte update and 2,822,400 target pixels each frame; unchanged source supplies 14 additional fixed bind/state calls and one update. Upload bytes are the existing logical ledger footprint, not a complete driver transfer measurement; repeated city constants are represented by update-call counts. Clear/target/copy footprints are not fragment invocation counts. B/C add 111 scissor saves, sets and restores per warmed frame (21 in the first cached frame). The ledger build additionally performs 111 raster-state checks and 111 restoration getters; those diagnostics are absent in primary timings.

B/C captures are bit-identical in RGBA and D24 at both hours; the pictures are deliberately blank. D deletes both reflected pass ledgers. Private normal versus frozen C7 has exact D24 and only 106/101 changed color pixels (maximum channel error 12/7), Frozen-C7 repeats show comparable 94/100 color pixels (maximum 11/7) with exact D24. These are bounded normal-reproduction checks; no blanket pixel equality is claimed. `diagnostic-contact.png` was inspected.

## One targeted material probe

E changes only main-ground `PSFeature` output for .5≤material.y<1.5, preserving original coastline sampling/alpha/clip, vertices, depth, bindings and all reflected/other-material shading. Corrected noon A/E then E/A pairs use the same 180-frame protocol with counters off. Warmed paired means are 138.795→126.905 and 134.259→124.918 ms; aggregate A/E is 136.527/125.912 ms, medians 133.722/123.982 ms. All 354 warmed frames in either arm miss 16.67 ms. Raw complete/transition/CPU distributions remain in `material-frames.csv` and `material-summary.json`. E’s counter fixture has identical original API/work rows to A, 763,592 changed color pixels and exact D24. Its delta includes possible downstream color/bloom interactions and is a bounded diagnostic opportunity, not a production gain.

The initial probe accidentally changed unused `PSMain`; those four runs and shader source are preserved as inactive-entry controls and excluded from E results. `Natural::load` actually compiles `PSFeature`; a focused regression now checks that contract. There was one target class, no effects sweep.

The initial recommendation was to specialize exact biome endpoints in the main-ground shader. Independent review defers that change: it can recover only part of the measured ~10 ms opportunity, while over 100 ms remains to be removed. It remains a valid smaller candidate, with coastline, mixed weights, derivatives and texture LOD requiring care. Reflection omission saves only 13–16 ms here; reflection coverage bins also wait. Native64 wave indexing remains a separate existing cap failure; no cap or truncation changed.

## Independent review and revised next assignment

Independent review reproduces all 2,880 primary and 720 corrected material raw-frame rows, full/warmed distributions, exact deadline counts, source clock and work/API invariants. All 1,043 evidence hashes, 194 runtime/link files in each of three source snapshots, and four binary identities match. Five independently recomputed full color/D24 comparisons reproduce the report; the contact sheet was inspected. Four focused tests pass on rerun, including real D3D11 scissor/state restoration. The receipt is `.cache/redraw-causal-step/auditor-review.json`. The separate-instrumentation wording above is the only reporting correction. This accepts the diagnostic conclusions, not a production speedup.

**Next assignment: test terrain geometry density before investing in smaller shader changes.** The default `PatchDetail` uses 64 subdivisions per side for detailed terrain; flat interiors already use at most 16, with matching dense edges. Existing `C3X_RENDERER_PATCH_PIXELS` controls can select coarser lattices. Reuse them for a short, reversed comparison against C7 at 2240×1260, keeping full materials, effects, actors, projection and clock. Verify the actual lattice and geometry submitted by main and reflected passes; do not infer activation from the environment variable alone.

The prior submitted-work ledger has about 1.76 million main natural-terrain triangles, 798,000 natural-decal triangles and 665,000 natural-mountain triangles. Similar terrain is submitted for reflected materials. These counts suggest a representation problem but do not establish its timing cost. The inspected 0 A.D. base heightfield uses two indexed triangles per cell in retained 16×16-cell patches, plus material-specific transition geometry. Its cell scale differs from Civ III's: compare projected error and useful screen coverage, not tile counts.

Test the existing 32- and 16-subdivision candidates first, with a normal control on both sides of the sequence. Proceed to one quality-preserving terrain representation candidate only if complete-frame savings justify it; roughly 20 ms or 15% is the prioritization guide, not a claimed speedup or a new acceptance target. Inspect early color/depth differences before developing a general LOD system. Preserve relief silhouettes, coast/river boundaries, material transitions, decals, shadows, reflections and wrapped edges; retain fine geometry where needed. Canonical immutable owners and bounded detail variants must survive camera changes. If reduced base density has little benefit, stop this branch and report that result rather than polishing it or silently switching to another optimization. Decal representation remains a separate candidate, since this control does not remove its roughly 800,000 main-pass triangles.

The goal remains 16.67 ms during supported actions, with full settled quality and only the permitted imperceptible or measured brief transition differences. A coarser diagnostic is not automatically a promotable candidate. Reuse existing harnesses and receipts, keep instrumentation out of primary timing, and avoid another broad telemetry project. No staging, installation, reference replacement or native ownership expansion is part of this assignment.

## Reproduction and limits

Use `python3 -m Renderer.tools.measure_redraw_causal prepare`, then `build --arm private` / `build --arm ledger`, and `run --label <unique> --arm A|B|C|D` (`--hour 12|1`, `--frames 180`). Private snapshots/build batches, shaders, source/API transformation manifests and binary hashes are retained under `Renderer/.cache/redraw-causal-step/`. The helper is never included by production; the build generator inserts it only into those snapshots. Existing output directories are preserved. Four focused tests pass, including real D3D11 indexed/instanced empty-color/depth and nested-state restoration.

One ledger compiler launch and the final night A launch failed in VM transport with no process started; failures and process checks remain, and the latter uses `primary-h1-b2-A-retry`. Initial compiler output-path, Windows macro and missing-export-file errors are preserved; successful compilation/link and ledger build receipts are recorded. No competing compiler/client/game ran during measurements.

The 47,224-file input closure and accepted C7 runtime/link sources match before and after. Original evidence is reused in place. Generated build intermediates and unused automatic initial images from this step alone are disposable after hash recording; final diagnostic captures, logs, DLLs, frozen sources and summaries remain. No seasons work, protected ignored inputs, earlier evidence, staging, installation, game session, references, injected files, wonders or District milestones were touched. Four synthetic actors and source-clock playback do not qualify native busy-game performance or sustained 60 FPS.
