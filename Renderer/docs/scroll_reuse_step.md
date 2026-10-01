# Two retained static raster states

Full static draws fall from 36 to 18 in each matched 18-view trace. The implemented candidate retains exactly two static regions, canonical 1x and one current display projection. World/content ownership, assets, shaders, selection and drawing code remain shared. A new non-1x zoom replaces the display identity while preserving canonical pixels and reusing display storage. The bounded change passed independent source, evidence and actor-crop review and is integrated into the main source branch as `37688c1f` (from `a519be23`); it is not staged or installed. Quality residuals and timing limits are recorded below.

`static_raster_state.h` owns the two noncopyable targets and their validity, revisions, projection, covered rectangle, anchors, actual depth translation/origin, scene/geometry, shadow and environment/light proof. Display allocation is lazy. Scene/wrap/classification/environment/light changes invalidate both states; layout/sample changes release both old targets. Phase, guard, depth-origin and anchor failures redraw the affected state. Full/strip writes retire validity before mutation, and failures preserve safe-redraw behavior.

`static_cache` remains one shared viewport target. Its last-writer proof includes slot, region revision, camera and actual depth; switching slots always restores even at unchanged camera. Shared reflection and reflected-terrain material targets have exact projection/generation/anchor/depth/light/shadow/extent/ROI writer keys and clear validity before writes. Reflections redraw on projection switches. Glow, bloom, aquatic targets, unit poses, streams, borders, transfer and publication work remain shared serialized scratch. Removal of reflected dynamic units restores the static base.

Canonical preparation still renders current-time water/resources/units and the remaining composition passes. Static reuse does not freeze animation or remove genuine scene-generation redraws. Exact projected shifts, fractional-phase rejection, guard checks and the clear-depth sentinel fix from the preceding scrolling change remain intact.

## Canonical consumers and future continuation seam

Preparation calls the fresh renderer at 1x and copies the completed canonical output into its owned publication. Adoption publishes that native source; projected display calls reuse the same world and pipeline. Native save/restore consumes canonical publications. A zoomed raster cannot replace the canonical publication or make a pending camera ready early.

The existing retired-camera/viewer/publication guards remain unchanged. If a camera retires before its first projected display, the completed canonical publication remains the fallback. Old published BGRA images own their lifetime independently of these HDR region caches; reflection binding still uses RAII.

Draws are currently synchronous under serialized immediate-context ownership. Future resumable work must hold the selected slot address/revision and immutable draw inputs through retirement, replace mutable `current_raster()` lookups with the held state, coordinate shared scratch writers, and prevent layout/reset/projection replacement from releasing or overwriting an unfinished job's targets. Capturing a slot alone does not make shared scratch concurrent.

## Counts and memory

The 18-view diagnostic at 2240x1260, native tiles 128x64 and one scene sample observes:

| State | Full draws | Viewport restores | Static reuses | Strip fills |
| --- | ---: | ---: | ---: | ---: |
| Canonical | 7 | 18 | 11 | 2 |
| Current display | 11 | 18 | 7 | 2 |

Canonical full draws cover cold start, two resident-generation changes, world content, environment and two depth-origin moves. Display adds two fractional 4/2 moves at 1.25x and two non-1x zoom replacements. World/environment/depth scenarios overlap scene/light/anchor/shadow causes; they do not isolate depth-only or local-light-only invalidation. The inactive display state is invalid before activation, then redraws once. Both slots reuse after each stable repeat. All 36 shared reflection draws rebuild; this patch retains static regions only.

Each 2888x1652 color/depth region costs 57,251,712 bytes (54.599 MiB). The measured GPU estimate rises from 238,221,152 to 295,472,864 bytes as the display region warms: exactly one region added. Both regions total 114,503,424 bytes; shared viewport storage remains 34,205,568 bytes. Views/aliases add no texture backing. Added-region arithmetic is 109.199 MiB at sample2 and 218.398 MiB at sample4; those are allocation estimates, not complete scene qualification at those settings.

Pass records account for submitted index/instance references and target/copy footprints, not shader invocations, actual shaded pixels, GPU time or scanout. Each stage snapshot covers the last fresh call, not every worker call in the ticket. Seed snapshots may repeat prior work and are excluded from totals.

## Quiet comparison

Four quiet copied-native runs use the same scene and common client, with buffered frame records and diagnostic/quality instrumentation disabled. Order is control noon, candidate noon, candidate night, control night. The table reports begin/readiness/adoption through display and Present return, excluding cold start; it does not measure physical first scanout.

| Hour | Motion | n per arm | Control mean / median ms | Candidate mean / median ms | Static full draws, control → candidate |
| --- | --- | ---: | ---: | ---: | ---: |
| Noon | Ordinary 4/2 | 2 | 438.21 / 438.21 | 306.89 / 306.89 | 4 → 2 |
| Night | Ordinary 4/2 | 2 | 332.29 / 332.29 | 291.81 / 291.81 | 4 → 2 |
| Noon | Integral-phase 8/4 | 3 | 213.63 / 112.94 | 114.78 / 107.39 | 6 → 0 |
| Night | Integral-phase 8/4 | 3 | 207.65 / 235.06 | 109.90 / 83.36 | 6 → 0 |

Across all 18 views in each hour, static full draws fall from 36 to 18. Canonical/display alternation no longer destroys the two retained regions. Ordinary 4/2 motion still forces display full draws because projected Y is fractional; 8/4 motion reaches integral reuse. Shared reflection and genuine generation changes retain their full passes.

These tiny reversed runs establish the structural reduction, not a statistically reliable latency, FPS or tail claim. All sampled ordinary/phase views exceed 16.667 ms. Other groups are mixed: noon stationary mean is 153.25 → 186.08 ms, and night zoom-replacement mean is 64.10 → 300.92 ms. Preparation, presentation stalls and genuine generation work remain visible; no blanket speedup is claimed.

The original quiet pilot and these static runs capture zero unit records and add zero synthetic actors. Diagnostic unit/reflected-unit triangles and submitted instances are zero; cities range from five to seven. This is standalone copied-native preparation/adoption evidence, not busy-unit capacity or installed-game qualification.

## Same-time full-redraw quality

The quality run compares 18 displayed views against independent forced-full draws at identical scripted time. Its oracle invalidates retained states, so subsequent warm history and timing differ from quiet/diagnostic runs. Alpha and stencil match in all 18 pairs. Fifteen pairs have 72–866 changed color pixels and at most one changed depth pixel; their largest single depth differences are 32,256 and 32,263 D24 codes.

The three integral-phase views have larger residuals:

| View | Changed color pixels | Mean absolute BGRA difference (0–255) | Changed D24 pixels | D24 pixels >1 code | Maximum D24 difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8/4 positive | 899,668 | 0.43244 | 308,127 | 54,098 | 133,682 |
| 8/4 continue | 899,114 | 0.43126 | 307,833 | 55,151 | 133,682 |
| Return | 66,479 | 0.02257 | 295,301 | 39,953 | 111,072 |

Native-pixel forest/river/coast/mountain/route/farm/city crops, high-outlier crops and full-frame review show no missing whole objects, coherent displacement, shifted coastline/routes or broad reversed occlusion in the inspected views. Fine shading/texture variation and sparse coverage-edge differences remain. The two forward views have mean RGB differences 0.57659/0.57502, with 0.9354%/0.9272% of pixels above ten channel levels and largest such connected components of 81/80 pixels.

Depth differences are not only quantization: pixels above 256 codes number 388/411/117; pixels above 65,536 codes number 22/23/4. Measured proximity places 99.998% of >1-code phase-view pixels within 3x3 of >1,000-code discontinuities; it does not establish the cause or correct fragment ownership. Odd projected Y shifts of five pixels may affect derivative-quad phase, but that contribution remains unproven. Exact image parity and dynamic-unit occlusion are not established by this fixture.

## Validation and evidence identity

The direct host test includes the production helper with fake noncopyable owning targets. It covers 100 alternating selections, display replacement, inactive invalidation in both directions, sample/layout retirement, exact target lifetime, stale writer revisions, partial region failure and failed cross-slot restore. Host validation reports three cases passing across `test_static_raster_state` and `test_scroll_region`.

A separate private diagnostic clone admits 16 actors through the existing `UnitInstances` body/state/motion owner and passes their poses to unchanged production `draw_real` calls. Every canonical/display/full triple reports 16 admitted, visible and travelling actors, 85 parts and the same 16 motion legs. Captured native unit records remain zero; these are admitted fixture actors. Display at 1.25x performs zero static full draws, one reuse and two strip fills in each positive/continue/negative phase view. Each display executes 170 main-unit draws / 62,942 triangles, 85 reflected-unit draws / 31,471 triangles and 170 self-shadow draws / 62,942 triangles. These count part/pass submissions, not actors or shader invocations.

The independent full oracle reuses the identical frame/camera/projection/time/world/light inputs and invalidates raster and shared reflection/material proofs without changing the unit owner. Within each pair the actor owner has the same ticks/frequency/parts/legs; no motion segment ends or new leg starts during the 165 ms trace. This establishes the same poses through the production owner's deterministic equal-time sampling and captured appearance; no per-pose byte hash was emitted. Native-size positive/continue/negative crops show moving actors partly hidden by vegetation consistently with full redraws, without missing whole actors, displacement, trails or broad occlusion reversal. A limited blue-clothing RGB heuristic matches exactly in 1,203/1,226/1,228 pixels in the three central crops; it is not a complete ownership mask. The corresponding >1 D24 counts remain 53,374/54,338/44,636, maximum 133,682, with zero alpha/stencil changes. Fine foliage/coverage differences remain; this short diagnostic establishes focused reuse/occlusion behavior, not long-gesture or busy-unit performance.

Four existing GPU contracts and two effect-config contracts pass: mixed unit scratch has 24 draws and two allocations with exact HDR/depth and 98,304 native resolve pixels; unit shadow has exact coverage and 221,488 finished pixels; retained composition passes 126 GPU oracles plus 120 independent-clock frames; projected display preserves detail, ordered overlays, fixed HUD, canonical images and retired-before-first-display behavior. Effect configuration defaults and failure propagation pass. No injected sources changed, so injected compilation was not needed. Receipts are in `focused-regressions-v2/receipt.json` and `runs/candidate-phase-actors-quality/actor-report.json`; the private fixture source and binary overrides are pinned separately and are not part of the production runtime candidate.

Frozen reviewed baseline is `4d82004864d882745699e5fd0a542ebae5638eec`, integrated baseline `dbf65855`. Production-recipe builds use `/std:c++17 /EHsc /O2 /W4 /WX`. Source snapshots, complete source/binary/shader/scene hashes and matching completion/transport receipts remain under `Renderer/.cache/static-raster-step/`; the six actor-free runs and separate actor regression have exit0 completion and pass transport receipts.

| Identity | SHA-256 |
| --- | --- |
| Control DLL | `9fb85b83681d5f6708e8094f5a4255075edfe8ab979811b1c01691c1d1b4041f` |
| Candidate DLL | `982a01b6280bf161af8be27cfbfe3502b816713d7f8455574e45669cbb236dd0` |
| Common quiet client | `e8b2cc2e476616153115d7189a60b3f93f135c8c7366775ab77f19702240e046` |
| Diagnostic/quality client | `1404583d3e9b07eeb45272ef32a71007fde2b32db02d4c9c5a86676ba0ab0ddf` |
| Developed copied scene | `22cd81c8be16ebe3a7aebc3ae33a00448f6afed6b6b945ced82b51b80bb139b7` |

Current compact evidence: `quiet-comparison.json`, `independent-diagnostic-audit.json`, `visual-review/assessment.json`, and each `runs/*/summary.json`. Historical duplicate `projection` diagnostic tokens are preserved in raw logs; corrected parsing distinguishes actual zoom from the reason counter. The two-state native runs used the same candidate production DLL; harness-only updates have their own source/binary receipts.

`vm-release.json` verifies all 17 owned launcher children completed with matching exit0 receipts and none of their PIDs remain. Renderer/compiler/game inventories are empty; no process termination was needed. Reservation `static-raster-validation-01` is released and no native work remains. The earlier failed focused-test adapter is preserved separately; it failed before VM dispatch.

No installation, reference replacement or injected-source changes occurred in this step. No new Civ III patch symbols are required; `required_user_action: none`. Use Git for implementation history; these notes describe the current measured candidate and its remaining limits.
