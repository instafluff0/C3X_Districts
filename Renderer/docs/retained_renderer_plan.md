# Retained renderer: current implementation and priorities

## Objective and boundaries

Complete the existing world → view → pass → publication path with bounded
ownership, full detail and source animation timing. Reduce repeated work, prepare
useful work before demand, and deliver it through Civ III's own calls. These are
three separate responsibilities and must be measured separately. Worker completion
never advances gameplay, notifies Civ III or requests a redraw.

The connected architecture instruction replaces the old single-experiment queue.
Preserve AGENTS.md, native overlays/visibility/picking, generic runtime assets,
accepted visual contracts and deferred wonders/Districts. Do not install or launch
Civ III as part of routine verification. Tested evaluation staging is authorized;
the user's entry point is ordinary `INSTALL.bat`, without environment settings.

## Actual code responsibilities

Completion means ownership and behavior for the admitted map categories, not
permanent GPU residency or knowledge of uncaptured gameplay.

| Responsibility | Implemented ownership | Remaining limits |
| --- | --- | --- |
| Persistent world and instances | `CapturedScene` retains canonical appearance, revisions and bindings separately from current observations. `ResidentContent` and shared mesh/asset owners supply reusable content; camera departure and GPU eviction preserve identity. | Captured surroundings only; bounded metadata admission can fail. Native unit bodies retain a separate bounded playback/pose owner. |
| Local validity | Compiled tiers and instance hits validate appearance, neighboring city exclusions, semantic neighbors, coast/world samples and river query-cell proofs. Native labels/population/selectors do not invalidate static geometry. | Device/asset/world-basis changes remain global. Interruption must discard partial assemblies without destroying valid content. |
| World → view selection | Compilation publishes handles. Current authoritative occurrences assemble passes; a spatial contributor index selects static inputs with exact intersection and ordered deduplication. | The index is view-scoped; dynamic poses use small scans. Native unit bodies arrive individually rather than as a complete active-unit pass. |
| Compatible submission | Shared meshes, tree instances, ordered materials, common depth and bounded shadow-page batches feed the same foreground/background map path. | Units still submit demanded misses individually. Optional reflection and other profiles retain their existing execution. |
| GPU reuse and output | Persistent static margin, viewport dynamic surfaces, sparse restore, incremental finishing, hardware resolve and consolidated readback serve the tested city profile with waves/reflections off. | Other profiles retain existing execution. Unit prediction batches at most two finished poses through one staging readback; demanded misses are not a consolidated pass. |
| Preparation and native delivery | Bounded CPU helpers prepare content/poses; one GPU owner prepares nearby static coverage, future map states and unit pixels. Copied snapshots, exact validity and pull publication connect to native calls. GOG has a gated 33 ms unit visual opportunity and 66 ms original callback deadline. | General nonblocking scrolling and complete active-unit demand collection remain integration work. Camera changes are immediate with exact demand fallback. Actual displayed FPS and interaction correctness are not established by callback counts. |

The native adapter uses the existing timer/Animator entry points, preserves the
elapsed accumulator and visual gate, and skips gameplay advancement on intermediate
visual calls. Map animation sampling stays at 15 Hz. Selected idle/work loops use
authored frames/duration; inactive unselected units freeze. Directed actions keep
native cursors and anchors. No copied animator, second presenter or camera swapping.
Exact GOG capabilities, guards and fallbacks: [patch ledger](civ3_patch_dependency_ledger.md).

## Latest live evidence and connected correction

The September 14 user trace contains 3,612 unit draws, 767 map completions and
518 intermediate visual refreshes. Faster cadence is active; this is not an FPS
measurement or an input-to-display comparison. Detailed debugger logging perturbs
latency, and phase percentiles describe the captured session rather than a matched
control. Only 26 complete DLL usage pairs were captured; do not generalize their
camera/idle distributions to an acceptance benchmark.

| Logged work | Observation |
| --- | --- |
| Unit draws | 97.3% pose hits; median 0.750 ms, p95 1.379 ms across all draws. The 97 misses have median 64.482 ms, p95 160.616 ms. Hits can still include a preparation wait. |
| Cold demanded poses | Median CPU pose phase 38.811 ms. Cache efficiency does not remove first-use preparation or its queue wait. |
| Prepared map publication | 572 consumptions; median wait 0.372 ms, p95 0.626 ms. |
| Map completions | Median 9.443 ms, p95 18.637 ms; initial 5.698 s load and a later 5.020 s rebuild dominate the maximum. |
| Speculative unit jobs | 125 jobs produced pixels; 633 produced none. Near one 203.829 ms Worker request, two jobs took 55.443 and 81.623 ms before demand ran. |

Two ownership defects are corrected in the current candidate:

1. A waiting unit draw or drain reserves the next GPU turn under the queue mutex.
   Cancellation alone previously allowed the worker to admit another optional job
   before the caller reacquired the mutex. Existing `camera_paused` is the worker's
   admission gate; this does not pause or modify the native camera. Cached CPU
   publications remain usable independently.
2. Recursive cliff cancellation has a distinct exception type. Camera cancellation
   now retires its partial draw assembly and preserves resident content, matching
   the ordinary cancelled return. Real exceptions still reset and now log their
   message. The live five-second rebuild followed a runtime exception and device
   reload at an unchanged view/world; the old cancellation path demonstrably causes
   such resets, but that trace lacks the exception message to prove this was its cause.

No assets, animation timing, detail, raster policy, worker counts or memory budgets
change. Submitted GPU work remains non-preemptible; demand can still wait for the
current speculative job. Both regressions fail with the previous behavior and pass
with the fixes, including reset/drain and real-error fallback. Native sources and
patch addresses are unchanged.

Validation: production Windows build, 22 targeted contracts, the actual worker
contract under MSVC/x86, and day/night native-unit/underlay/terrain replays pass.
Both controls pass four topology and six appearance edits against independent
redraws. All 146 busy/idle comparison images and 12 diagnostic images are exact;
both 14-step dense-scroll runs pass exact revisits. No native source changed,
so a new injected compile is unnecessary. No game install/launch or reference change.

| Whole-request workload | Control mean / p95 | Candidate mean / p95 |
| --- | --- | --- |
| Busy eight-unit mixed actions, 80 requests each across reversed run order | 240.02 / 293.41 ms | 135.77 / 174.74 ms |
| Warm realistic idle, 60 requests each, 67 ms caller pacing | 4.02 / 6.20 ms | 3.78 / 5.20 ms |
| Dense scroll, 14 offsets each | 241.58 / 598.29 ms | 248.44 / 591.69 ms |

The busy comparison improves mean whole-request time by 43.4%; warm idle and dense
scrolling show no substantial demonstrated gain. Two visible terrain edits average
651.27 → 628.89 ms, too few samples for a speed claim. These are 1119×1192 harness
requests with the existing 15 Hz animation clock, not native displayed FPS.
Whole idle requests include map, output copy and all eight bodies; warmup and saved
BMP writes are outside the interval. Cold initialization remains separate.

A bounded diagnostic (10 warmup plus 10 measured frames, 160 body draws each)
resolves the phase shift: speculative finished poses fall 133 → 26 while demanded
misses rise 36 → 135. Total finished builds fall only 169 → 161. CPU helper
consumptions at the last logged snapshots rise 31 → 121. The measured gain is mainly
better scheduling and useful concurrent preparation, not 43% less rendering work.
In the timing runs, mean map-phase cost falls 212.69 → 55.27 ms while unit cost rises
27.20 → 80.36 ms; judging either phase alone would misrepresent the result.
Diagnostic timings are excluded from the performance comparison.

The tested candidate is staged for ordinary `INSTALL.bat`; DLL SHA-256
`dd83f651e84101831d02a058878588ce1b1f7a5161bceed1f7c9281e2cb3f165`.
Local evidence, invocation manifests, immutable control/candidate binaries and
receipts: `Renderer/native/build/demand-priority/`.

## Next architectural step

Finish demanded-unit pass preparation: collect the eligible active poses at a
verified native visual boundary, resolve exact content keys once, prepare missing
CPU poses concurrently, then feed compatible GPU submissions and consolidated
readback. Reuse existing playback, caches and submission owners; the native body
call remains the final authority for placement, underlay and exact fallback.
Establish a clean collector before adding more workers or extending prediction.
If no narrow collector exists, retain the current call/return boundary.

Evaluate total frame/request cost, cold actions and sustained demand alongside warm
idle, dense scrolling and local edits. Track wasted preparation and interruption
separately. Do not claim a speedup from cache hits or shorter caller waits alone.
Resolve any further full-reset cause from the new exception diagnostic before
expanding GPU speculation. Native presentation/overlays/picking remain a strategic
integration checkpoint; the user's current log is reused rather than requested again.

## Preserved controls and closed findings

- Pre-change source: `a504a106`; staged control DLL SHA-256
  `549daee21744e6724ae9b5080cb2b14714e9f89cab9d663c63411a9c4abca028`.
  Exact control binaries are retained in `native/build/demand-priority/control/`.
- [Earlier checkpoints](history/retained_renderer_checkpoints_20260914.md) preserve
  world/view, shared meshes, neighborhood preparation, native handoff, animation,
  receipts, accepted differences and superseded implementations.
- [Completed output record](history/retained_output_completed_20260913.md): dense
  comparison 119.40 → 103.28 ms; stationary work did not improve. Do not restart
  output-helper or rejected resolve experiments without a different mechanism.
- Global 30 Hz map sampling was rejected: warm whole requests 3.970 → 7.462 ms,
  p95 4.974 → 22.680 ms, all 62 images exact. It is not part of this candidate.
  Evidence: `native/build/native-visual-cadence/`.
- Parallels GPU timestamps/event-query attribution remains unreliable. Whole-request
  CPU wall time is the performance authority; readback includes queued GPU work.
