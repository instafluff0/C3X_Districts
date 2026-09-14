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
| Preparation and native delivery | Bounded CPU helpers prepare content/poses; one GPU owner prepares nearby static coverage, future map states and unit pixels. Copied snapshots, exact validity and pull publication connect to native calls. GOG has a gated 33 ms unit visual opportunity and 66 ms original callback deadline. | Prepared nearby cameras now acquire exact-current-camera crops without GPU waits. Cold/outside-area scrolling and complete active-unit demand collection remain integration work. Camera changes stay immediate with exact demand fallback. Actual displayed FPS and interaction correctness are not established by callback counts. |

The native adapter uses the existing timer/Animator entry points, preserves the
elapsed accumulator and visual gate, and skips gameplay advancement on intermediate
visual calls. Map animation sampling stays at 15 Hz. Selected idle/work loops use
authored frames/duration; inactive unselected units freeze. Directed actions keep
native cursors and anchors. No copied animator, second presenter or camera swapping.
Exact GOG capabilities, guards and fallbacks: [patch ledger](civ3_patch_dependency_ledger.md).

## Latest live evidence and current implementation

The latest user trace has one initial device load, no mid-session reload and no
camera-error/cancellation records. Its 2,007 unit draws include 1,922 pose hits
(95.8%); the 85 misses have median 60.759 ms and p95 127.338 ms. Prepared map
consumption is already quick: 269 returns, median 0.369 ms, p95 0.663 ms.
Of only 24 complete DLL usage pairs, 13 nearby-camera requests have median
156.315 ms and maximum 359.497 ms. Ten stationary requests have median 35.932 ms
and maximum 243.143 ms. These debugger observations prioritize remaining stalls;
they are neither matched benchmarks nor displayed FPS. The earlier ownership fix
and its measured 43.4% busy-request gain remain in the checkpoint history.

The current step is caller-driven nonblocking nearby scrolling. A fresh native
capture must acquire full-detail pixels at the **actual current camera**; otherwise
it uses the existing exact synchronous fallback. No old-camera substitution,
notification, native terrain fallback or new patch symbol is introduced.

Implemented design:

- One committed surrounding area and one in-flight replacement use the existing
  GPU owner, passes, demand priority and cancellation gate. Foreground and
  background use the same working extent; small camera changes leave its world
  placement fixed. Existing compiled content, static color/depth and incremental
  output remain reusable across animation refreshes.
- Native viewport geometry/depth units and existing detail tiers remain separate
  from working dimensions. Padding is at most 128 pixels per axis within
  2240×1192. Each finished CPU area, including its ready centered view, is capped
  at 32 MiB; existing front/snapshot/temporary allocations are additional.
- Fresh capture validates complete topology, identity/visibility, local dependency
  content, projection/detail and anchor translation. Only full captured appearance
  grants newly visible ownership. Current occurrences receive remapped flags.
  An identical centered capture leases its already finished image directly.
- The optional preparation export follows successful native composition. It copies
  input before ownership merging and replaces redundant native ambient queueing.
  No new patch address or callback is introduced. Unknown content, ambiguous wrapped
  occurrences, unsupported dimensions and failed preparation retain exact fallback.
- Static validity is independent of ambient sample freshness. A delayed producer
  may hold its honest old map-animation sample while the camera advances through
  valid coverage; authored playback catches up using the caller's clock. Native
  unit animation, overlays, visibility and picking retain their existing owners.

## Validation and measured effects

Windows production build, 50 focused contracts, the actual worker under MSVC/x86,
injected compilation and day/night unit/underlay replays pass. Nine prepared
cameras match an independent reconstruction of the same prepared area exactly.
Four topology and six appearance edits match independent redraws. The wider
14-step sequence passes exact camera revisits, with no fallback tiles or recovery.

| Whole consumer request | Control mean / p95 | Candidate mean / p95 |
| --- | --- | --- |
| Nearby real-clock scrolling, 48 calls, 67 ms opportunities | 55.31 / 79.17 ms | 11.45 / 14.57 ms |
| Warm realistic idle, eight bodies, 60 calls | 3.76 / 5.28 ms | 4.54 / 5.13 ms |
| Wider 14-step scrolling with coverage misses | 219.18 / 479.02 ms | 216.66 / 387.27 ms |

These are 1119×900 harness capture/render/copy requests, not displayed game FPS.
All 48 nearby calls acquired prepared current-camera pixels; oldest map sample
was 183 ms. Initial dense rendering increases 6.04 → 7.35 s. Warm idle has a
0.78 ms mean overhead and no demonstrated speedup; wider scrolling shows no
substantial mean gain. Native unit body work is included in the idle endpoint.

A separate bounded diagnostic distinguishes work reduction from off-thread work:
48 → 44 map jobs, 75,326 → 3,474 static selected submissions including guard fill,
and all 44 candidate surfaces reuse static color/depth. Geometry builds/uploads
are zero in both. Dynamic selections increase 3,376 → 3,696 and resolved-pixel
accounting rises 192 → 283 million. Candidate area jobs total 1.288 s including
CPU publication, while control map-render intervals alone total 1.794 s. These
instrumented wall intervals explain the mechanism, not an independent GPU timing
claim. Maximum observed area ownership is 15.32 MiB; recorded free contiguous
address space remains at least 1.60 GiB in the x86 harness, not a live-game guarantee.

**Visual review remains pending.** Full detail is retained, but the new working
surface changes 8,028 of 1,007,100 pixels (0.80%) against the control; maximum channel
delta is 126, mean absolute channel delta 0.042/255. Independent prepared builds
and camera revisits agree. This is new raster variation, not covered by earlier
acceptance. [Local comparison](../native/build/prepared-scroll/visual-comparison.png).
No reference was replaced. The exact tested DLL is staged for evaluation;
installation and launch remain the user's actions through ordinary `INSTALL.bat`.

Local receipts, binaries, logs and invocation manifests are in
`Renderer/native/build/prepared-scroll/`. Candidate SHA-256:
`4cbdfa93b44a7f41d18280db3932d216dc794ece15d08a3dfd1cfdcb3d85e8be`.

## Next responsibility

Validate actual GOG camera cadence, overlays/picking and unit interactions at this
strategic integration checkpoint. General coverage is still bounded: a cold view,
content change or jump outside prepared coverage can block. The next architectural
extension should broaden resident static coverage while selecting a narrower
animated/output working set; the diagnostic shows why simply animating a larger
area or adding workers is not sufficient. Complete active-unit demand collection
remains separate work, supported by the live trace's expensive uncached poses.

## Preserved controls and closed findings

- Current reproducible control: `9c5e9033`, DLL
  `dd83f651e84101831d02a058878588ce1b1f7a5161bceed1f7c9281e2cb3f165`,
  preserved in `native/build/prepared-scroll/control/`.
- Earlier source: `a504a106`; staged control DLL SHA-256
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
