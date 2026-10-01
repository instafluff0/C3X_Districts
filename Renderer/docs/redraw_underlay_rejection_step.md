# Actual underlay mask and early rejection

The one private correction saves **15.829 ms (11.36%) at noon**, **22.243 ms (14.70%) at night**, and **21.618 ms (16.15%) during fixed-1.25× pan**, including the stencil/prepass costs and every measured stall. It adds no useful benefit to retained 1× pan. This is a useful full-render cost reduction, still far from 16.67 ms. **It remains unpromoted for Astra's review:** the temporal sweep contains one non-repeatable D24 pixel exception, alongside sparse color differences described below. No staging, installation, reference replacement or injected-source change occurred.

This closes the bounded follow-up to [the previous underlay step](redraw_underlay_step.md). All evidence is under `Renderer/.cache/redraw-underlay-rejection-step/`; accepted inputs and the prior evidence were reused and verified unchanged. The following results concern the frozen full-density C7 native128 fixture on the Parallels adapter, not a live-game or many-unit qualification.

## Mechanism and one correction

The diagnostic copied the **actual static-region D24S8 immediately after stage2 and before stage3**, with separate before/underlay/mask snapshots. Stage1 uses the admitted original underlay geometry/VS, no PS, and original depth writes. Its raster/depth result defines the denominator; the whole padded target does not.

| Projection zoom | Underlay depth-covered pixels | Marked intersecting pixels | Marked fraction | Marks outside underlay |
| --- | ---: | ---: | ---: | ---: |
| 1.000000 | 2,850,464 | 1,041,327 | 36.53% | 0 |
| 1.122222 | 2,850,464 | 1,155,719 | 40.54% | 0 |
| 1.250000 | 2,850,464 | 1,220,939 | 42.83% | 0 |

The target is 2888×1652, R24G8_TYPELESS, sample count1; viewport covers that target and scissor is `(320,192)-(2568,1460)`. Stage2 leaves D24 exactly unchanged. Bound state/shader/target identities match the intended stages: stage2 is LEQUAL, no depth writes, S8 read/write bit1, ALWAYS/REPLACE, reference1; stage3 is LEQUAL, depth writes, S8 read1/write0, EQUAL/KEEP, reference0. Both coverage shaders and the shaded-underlay entry match the intended bound PS. Target identity is continuous through the captures. The logged VB stride precedes the draw-record binding; it is not evidence of that draw's final vertex stride.

The mask keeps the real coast/alpha guards. `lab/shared/natural/ground.h` writes `world_valid=1+coast_ramp((distance-beach_width-.10)/.90)`, the packer preserves it, and city terrain derives `coast_inland=max(0,world.w-1)`. Coverage requires inland1 and alpha1 using the original texture/grain expression. No epsilon was relaxed to increase coverage. `mask-summary.json`, the twelve lossless raw target snapshots, and `mask-contact.png` retain the measurements and their spatial relationship to shore/river gaps.

The sole shading correction appends `[earlydepthstencil]` to `PSSandboxUnderlay`, whose dedicated entry forces kind0.5. The admitted compiler creates panel-alpha1 underlay vertices, and the active city-profile shading path returns alpha1, with no PS depth output or UAV effects. The actual compiled shader retains one final alpha discard instruction, dormant on this admitted opaque path; it is not claimed to be globally discard-free. The clipping coverage entries remain late. [Microsoft documents the attribute's early-test behavior](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-attributes-earlydepthstencil).

Actual old/new disassembly is identical after removing the added `forceEarlyDepthStencil` global flag and its comment: 2202 instruction/declaration lines, one final discard, no PS-depth output, no UAV, 82,036-byte blobs. `shader-flag-verification.json` and the two disassemblies record this. The corrected blob is `compiled-evidence/hydrology.hlsl.PSSandboxUnderlay.4a56f8ce6409d23a.cso` (SHA256 `be8193d70103661d980d8099528d5851cfd9e5b57f506effc58495af06c6be0d`).

The same candidate also implements the explicitly required empty-underlay guard **before all extra work**, removing the unnecessary water-pass stencil clear. Draw/blend/final-depth order and conservative partial coverage remain unchanged. Diagnostics are disabled in the primary runs and allocate no staging textures or forced-control state there. The correction is generated in a private source snapshot by `tools/measure_underlay_rejection.py`; production `sandbox/fresh_pipeline.h` is unchanged.

## Causal controls and full-frame measurements

Always-pass/always-fail controls keep the same geometry and prepasses. Each pair ran twice in reversed order, 46 source frames per run; the warmed pool has86 samples per arm. These outputs are deliberately invalid for fidelity.

| Compiled underlay | Always-pass mean | Always-fail mean |
| --- | ---: | ---: |
| Original dedicated entry | 141.605 ms | 144.722 ms |
| Forced-early entry | 144.493 ms | 84.553 ms |

This establishes useful rejection behavior for the corrected draw on this adapter. It does not measure GPU invocation counts, isolate an additive GPU duration, or prove exactly how the adapter scheduled the original shader. No timestamp or pipeline-statistics queries were used.

Quiet measurements use actual2240×1260/60Hz, Present1, explicit `time_ms=29533+frame*33`, native128 capture and the same actors/assets/shader roots. Each cohort has two reversed pairs of180 frames. Capture, counts, trace, completion probes, queries and causal controls are off. Primary zoom uses the original frozen client. A separate frozen QA client supplies two repeated90-frame pan cycles, from `(0,0)` to `(90,45)` and back, at fixed1× or1.25×; its parity check reproduces every pass/owner field and D24, with95 sparse color pixels, max7. It does not certify arbitrary prolonged pan or a live-game capture cadence.

Warmed distributions omit only the protocol's first three frames (354 samples per arm). All remaining stalls stay in the distribution. Values below are milliseconds, `mean / p50 / p95 / p99 / worst`.

| Cohort | C7 control | Corrected candidate | Mean saved |
| --- | --- | --- | ---: |
| Noon zoom | 139.376 / 140.165 / 146.803 / 269.820 / 451.196 | 123.546 / 118.375 / 125.048 / 280.486 / 1053.867 | 15.829 |
| Night zoom | 151.280 / 150.379 / 162.574 / 336.929 / 426.632 | 129.037 / 127.846 / 137.586 / 258.682 / 403.923 | 22.243 |
| Pan 1× | 20.541 / 20.468 / 21.893 / 31.745 / 46.766 | 20.918 / 20.361 / 21.786 / 45.527 / 122.542 | -0.376 |
| Pan 1.25× | 133.875 / 133.416 / 137.739 / 274.210 / 379.567 | 112.257 / 112.207 / 116.010 / 218.170 / 311.858 | 21.618 |

Both blocks improve zoom and 1.25× pan: noon saves21.928/9.730 ms, night23.958/20.528 ms, pan1.25×20.866/22.370 ms. Noon block2 includes two candidate Present stalls: frame61 total1008.155 ms (Present990.839), frame62 total1053.867 ms (Present1036.069). The typical noon median improves21.791 ms; those stalls limit its inclusive mean gain. They have not been excused or attributed to a particular cause.

Complete180-frame pooled means are noon139.747→123.019, night150.577→128.000, pan1×22.678→22.248, pan1.25×133.570→112.434. Complete/warmed total, draw and Present distributions, every per-run result and all2880 primary frames are in `primary-summary.json` and `all-primary-frames.csv`. Every warmed zoom/1.25×-pan frame misses16.67 ms; candidate1× pan misses342/354. Even the best expensive cohort remains95.590 ms above budget on its mean.

## Appearance, depth and transitions

Focused executable tests pass24 cases of partial/opaque coverage, replacement in front/equal/behind and later coplanar overlays, with direct S8 checks. The compiled early flag is asserted on the opaque entry and absent on the clipping mask. An additional48 WARP comparisons check every RGBA/depth sample in the **unmasked 2×/4× MSAA fallback**, where the prepasses are disabled. These synthetic MSAA tests do not certify a full-scene multisample adapter mode. The existing19 capture/ownership/zoom/config-off tests also pass; their runtime sources are unchanged.

Full captures are2248×1268 (including the renderer's guard pixels). Every owner CSV field and duplicate multiplicity matches between arms: 2269 rows per zoom capture, and all thirteen adopted views. The seven same-clock checkpoints per noon/night run have exact D24. Color differences are sparse: noon max33, at most12 pixels over8, worst mean RGB error0.000674 in eight-bit units; night max26, at most2 pixels over8, worst mean0.000032. These are measurements, not reference replacement or acceptance.

Thirteen views cover origin, diagonal, newly exposed strip, equivalent occurrence, both wraps, unrelated jump, zoom settle, zoom pan, return, and resident strip/wrap/jump. All corresponding D24 and owner fields match. Maximum view color error is51 on wrap; that view has7040 changed pixels,210 over8, mean RGB error0.002783. The other view metrics are retained in `views-quality-work.json`; they are not collapsed into an exact-color claim.

The temporal sweep preserves sixty full color/D24 sample pairs, every third source frame, plus a paired movie at10.101 Hz over5.94 source seconds (`normal-speed-comparison.mp4`). The paired five-frame contact sheet and the outlier crop were visually inspected. The timeline advances and reverses zoom while animated scene effects continue. Readback-run timings are excluded from performance/cadence claims; this is source-speed fidelity evidence, not certified60Hz display delivery.

**One temporal D24 exception remains recorded:** frame141 has one changed pixel at `(448,831)`, control6325167 versus candidate6353308 (difference28141/0xffffff); that pixel's RGBA is identical. The frame also contains a separate sparse color outlier, max123. All other59 temporal sample depths match exactly. Repeating the identical180-frame trace with control and candidate, capturing frame141 again, yields exact D24 and max7 color difference between the repeats; the earlier candidate exception does not recur. `outlier-repeats.json` retains all six comparisons. This is a non-repeatable exception, not proof that the original failure was harmless or a baseline depth fluctuation. Astra should review it before promotion; no threshold or shader variant was added to hide it.

Cold/adoption costs remain material and are separate from warmed timings. Across quiet zoom runs, client preparation is6.219–8.084 s for C7 and6.513–8.227 s for the candidate; first correct native128 view is3.033–3.217 s versus3.111–3.496 s. First-three zoom frames range9.694–357.931 ms versus9.836–255.066 ms. A thirteen-view diagnostic has origin3.480→3.288 s, new strip234.902→215.989 ms, unrelated jump2.153→2.029 s, wrap106.515→126.676 ms and zoom settle121.968→102.306 ms. These single adoption comparisons are observations, not causal cold-start speedups. Preparation/adoption/geometry/draw/upload fields for every view remain in receipts and `adoptions.json`.

## Added work, preservation and stop

Warmed noon/night submitted work reproduces the same increment: **2,813,721 triangles,1082.458 draws and277,109 upload bytes per frame**, including both prepasses. Total triangles10,049,496→12,863,217; draws5176.153→6258.610; uploads2,178,052→2,455,161 bytes. Logical target/clear footprint28,734,772→33,505,748 (+4,770,976 pixels). Copies remain3,659,240 pixels, rebuilds2/reuses1. The empty-water guard removes the previous candidate's additional2,850,464-pixel clear. Counts are logical submitted work, not GPU fragment counts.

The mask reuses existing S8; no additional render target, scene owner or geometry cache is allocated. It retains the previous candidate's two depth-stencil states, no-color blend state and two cheap coverage PS entries. Diagnostics additionally own staging/forced-state resources only when enabled. World-owner accounting remains2269 owners,407,725,211 cache bytes,338,461,196 unique GPU bytes,156,392 retired/selection bytes,63 built/1040 reused. Adapter-reported budget/usage is not physical VRAM certification.

`preservation-verification.json` verifies all47,224 accepted inputs (12,238,666,273 bytes),193 runtime sources, all823 prior evidence files,196 probe/196 corrected/217 QA-client source files, five binary identities, actual shader roots and the staged bridge/DLL/helper tuple. It verifies38 successful receipts and separately retains the two resolved directory-launch failures. All384 retained lossless scene/mask archives decompress to their recorded hashes. Fifty-two owned generated intermediates (unused automatic startup pictures and successful-build objects,347,372,591 bytes) were retired only after verification; their receipts are retained. Full witnessed color/D24, mask targets, source snapshots, binaries and failed build/launch logs remain. Compiler/test clients and Civ III are closed.

This bounded correction and qualification are complete. Recommend reviewing the measured private candidate and its fidelity exception before any production promotion. No second mask design, density tuning, unrelated shader change, ownership change or deferred milestone was attempted. The user's retained-image zoom-preview direction is a subsequent interaction assignment; it was not implemented during this slice.

## Independent review and next assignment

Independent review reproduced all 2,880 primary timings and distributions,
93 color/depth comparisons, ownership multiplicity, three actual mask captures
and their bound state. It verified 1,275 evidence files, 193 runtime sources,
the three private source snapshots, five binaries and actual shader roots.
The compiled shader differs only by the early-test flag. Both focused tests
passed again in 7.777 seconds, covering 24 mask cases and 48 MSAA cases.
The receipt is `auditor-review.json` beside the evidence.

The measured private speedup is supported. The temporal depth exception and
long Present stalls remain unexplained; preserve the candidate unpromoted.
No further underlay iteration is assigned now. Next: implement and measure a
temporary retained-image zoom preview on accepted C7, with full-quality
refinement and real wall-clock input. Roughly 124–129 ms remains the candidate's
full zoom redraw cost; the 16–22 ms reduction is not a 60 FPS result.
