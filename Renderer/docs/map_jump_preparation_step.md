# Map-jump preparation: cooperative FRESH cancellation

This source/host slice adds cancellation checks around the existing canonical FRESH preparation callback. A superseded destination can skip work not yet issued and cannot commit output or latch a renderer failure after cancellation. Resident content and native publication ownership remain unchanged. Native timing/build validation is pending; no camera-to-display or FPS gain is claimed.

## Actual path and repeated work

The native composition owner copies the authoritative capture, submits a camera ticket, polls readiness and commits only the matching completed view. `Navigation` keeps a completed fallback while the next capture is pending. `RendererWorker` keeps one active snapshot and one replaceable pending snapshot; a newer begin marks the active job cancelled. Helper IPC coalesces queued camera requests; a transport call already executing remains single-flight.

`publish_scene_capture` adds copied observations to the persistent journal. `WorldTopology`/`WorldCoast`, observation semantics and river proofs validate the same resident content. The compiler key omits camera anchors and animation time; canonical native-128 construction normalizes equivalent wraps. Valid `compiled_views` are selected before scheduling missing jobs. `ContentPreparation` retains completed owned results, drops duplicate active jobs, and lets an obsolete foreground `take` stop joining within its 1 ms polling boundary. Adoption still checks current dependencies before using a result. A geometry/selection epoch protects occurrences and pointer lifetimes; it does not prove world content changed.

`RendererState::render` then assembles current occurrences under immutable generation leases, prepares waves and the canonical native-size map, and returns metadata. The worker copies a completed canonical GPU publication only if cancellation remains false; ticket/atomic checks protect ready/adopt. Projected display consumes that owned publication and the accepted two-raster pipeline. Native save/restore and last-completed-view fallback remain authoritative.

Historical reviewed owner evidence distinguishes these cases:

| Selected-capture fixture visit | Newly built static owners | Raw upload bytes | Interpretation |
| --- | ---: | ---: | --- |
| First jump | 385 | 81,630,698 | New destination content in this fixture; 718 selected owners reused |
| Resident jump / origin / strip return | 0 | 0 | Existing immutable resources selected |
| Positive / negative equivalent wrap | 0 | 0 | Camera occurrence changed, content retained |
| Local appearance edit | 81 | 36,714,854 | One changed owner plus 80 dependency owners; 1,022 retained |
| Visibility change | 0 | 0 | New view/visibility does not rebuild unchanged static content |

One diagnostic first jump records 1,785.20 ms preparation wait, 1,737.93 ms geometry work and 32.06 ms canonical draw, followed by 1.05 ms adoption and 1,806.05 ms request-through-view completion. Its world queue has 385 scheduled jobs, 383 consumed combined results, two recoveries and 1,319.94 ms join time. Worker sums are concurrent CPU spans, not wall/GPU time. The paired prior resident-wrap means are roughly 85–97 ms readiness and 108–121 ms first-view submission with zero static upload. Those results predate the accepted two-raster patch and use source-grounded standalone captures with four synthetic actors; they are not installed busy-play qualification.

Ten historical superseded jobs end before logged wave/FRESH callback work. They verify retirement without identifying the frequency or cost of cancellation during FRESH. The recent scrolling request-through-Present-return means likewise measure small-fixture latency/work removal, not displayed cadence or proximity to 60 FPS. The targets remain 60 FPS for idle/scrolling and 40–50 FPS acceptable during zoom.

## Topology, appearance, backing and GPU readiness

| Ownership layer | What is already retained | Readiness limit |
| --- | --- | --- |
| Compact topology | Complete array, four bytes per actual tile (`width * height / 2`) | Terrain/river/activity data does not supply roads, objects, city/resource art or full appearance authority |
| Captured appearance | Persistent canonical observations from selected captures and accepted pages | Explored appearance must be captured; topology-only halo records cannot authorize omitted art |
| Compiled backing | Validated compressed compiler output under its existing bounded store | Backing must first be produced; after GPU eviction a hit still requires upload |
| GPU generations | Immutable resident resources and bounded selected/retired leases | Current active geometry ceiling is 2,048 MiB with admission, separate from assets, targets, queues and complete physical VRAM |

The historical first-visit fixture does **not** establish that installed whole-world preparation is absent. `WHOLE_WORLD=1` retains the complete CSV as fixture source inventory, but `reference_x64.cpp` reduces the initial frame with `SandboxCameraWitness.capture`. Later camera requests publish selected RENDER/PREFETCH appearances and TOPOLOGY_HALO records. The witness never registers or feeds world appearance pages and never waits for world-status readiness. The reviewed run has 14 scene publications, zero `world-input` and zero `world-region-prepared` records. Its origin/jump capture contains 1,983 records: 759 RENDER, 752 PREFETCH and 472 topology-only.

Installed injected setup registers `capture_custom_renderer_world_page`. The caller-thread capture timer accepts at most 128 records per page; a completed accepted pass increments `world_input.passes`. Background region preparation requires `passes > 0`, complete topology, valid scene/cache state and no foreground ownership. Each region independently requires current explored appearance/visibility authority. Foreground traffic can interrupt preparation; completed backing/GPU residency and budget eviction remain distinct states.

For this fixture, missing appearance-page bootstrap and unprepared selected destination explain the cold demand path; the jump consumes actual compiler work with no backing restores and no recorded owner eviction. Logs cannot say what an installed first visit costs after pages/regions finish, whether interrupted preparation or pressure leaves a real destination unready, or whether the whole explored world's detailed GPU resources fit. Record `world_status` authority, capture passes/cursor, region completion/unavailability, backing hits and current resident handles before drawing those conclusions. A 384/768 MiB or 32-bit geometry explanation does not describe the current Renderer64 2,048 MiB ceiling.

## Bounded change and ownership

The FRESH early return previously bypassed later generic cancellation checks. Its wave preparation, target/view creation, full callback and output commit could continue after new input; the worker correctly suppressed subsequent publication but had already paid that work.

The patch uses the existing `cancelled()` predicate before FRESH metadata/work, after wave preparation, immediately before the callback and after it before validity/output commit. Allocation/view failures give observed cancellation priority over the sticky failure latch. A created temporary RTV is released before the pre-callback early return and after the callback. Real wave/allocation/view/draw failures keep their existing failure behavior. There is no new owner, cache, allocation budget, worker, callback signature or static/dynamic stage change.

Already executing callbacks and submitted GPU commands cannot be preempted by these checks. Useful immutable world results survive through existing lifetime/retirement handling. The worker's current-ticket/atomic checks and `discard_scene_view` remain the publication authority. This closes a demonstrated control-flow gap, not a measured dominant jump bottleneck. Duplicate active-job input/snapshot preparation is a separate unmeasured finding; its scheduler already avoids duplicate compilation and it is deferred.

## Host validation and next native decision

The new portable test extracts the actual FRESH branch plus worker capture/commit expressions and runs them against recording D3D stubs. Nine cancellation cases cover before metadata, wave success/failure, texture/view success/failure and callback success/failure. It verifies expensive-call suppression where possible, no cancelled output commit/failure latch, unchanged completed/publication and resident-owner sentinels, and every created RTV's release. Normal success and four genuine failures preserve prior behavior. The same contract fails against the pre-change production branch. Six existing portable camera transaction, durable-input notification/lifetime, resident-generation, dependency/wrap and region-schedule contracts also pass. Fail-closed guards reject Windows/native dispatch, `prlctl`, `cmd` and shell processes before creation. An initial pair of test-name lookup errors is preserved; their corrected contracts pass.

One future granted native comparison is sufficient: matched control/candidate at full 2240x1260, native-128, sample1, normal reflection/water/waves and a genuinely busy destination with executed actor counts. Record complete topology, appearance-page and region/backing/resident readiness. Use one short first-visit/return/equivalent-wrap trace followed by rapid repeated A/B/C jumps ending at the latest requested destination. In separate diagnostic instrumentation, record cancellation arrival phase, callbacks issued/completed, skipped stages, upload/build/lease counts, ticket commits and fallback preservation; keep quiet newest-destination latency and presented-frame cadence separate. Count mid-callback cancellations as work that this patch cannot interrupt. Preserve exact source/binary/input hashes and owned-child cleanup. No expanded matrix or residency redesign is needed to decide this patch's native benefit.

Evidence stays in `Renderer/.cache/map-jump-preparation-step/`: source/content/readiness reviews, raw-evidence hashes and compact recalculation, executed host logs and expected negative-control receipt. Prior evidence is preserved. No VM calls, binary staging, installation, reference replacement or injected-source edits occurred. No new patch symbols are required; `required_user_action: none`.
