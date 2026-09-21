# Recorded renderer workload contract

The target is one approximately ten-minute gameplay capture that can drive the
**production renderer from its inputs**, be inspected at any recorded frame/time,
and be reused to accept or reject architectural changes. This is the evidence
contract for M4.0 in the [roadmap](retained_renderer_plan.md), not a separate
renderer or additional milestone ladder. The older composition journal does
not meet it. A separate [development input protocol](../native/input_recording/README.md)
is implemented but has not completed the qualification below. Do not ask for another long manual recording until the automated
capture/replay qualification below passes.

## What must be reproducible

Record from renderer startup, retaining the initialization prelude, and mark a
600-second gameplay interval after the first coherent map. Starting in the middle
of an existing game requires a separately proved complete checkpoint. A screenshot
or a page of visible tiles is not such a checkpoint.

Each frame needs a stable identity connecting input arrival, accepted scene/view,
visual sample time, production work, composition and presentation result. The
reviewer must be able to select second 327 or frame 8,123 and inspect that image,
the preceding changes, timing, memory and relevant native/renderer decisions.
Record native and ambient-only output: gameplay does not need a new map input for
water to move. Record opportunities that did not produce a frame, with reasons.

There are three distinct fidelity claims:

| Claim | Required evidence |
| --- | --- |
| Same renderer inputs and logical time | Complete initialization plus ordered immutable changes and causal dependencies, not pre-rendered map/pose pixels |
| Same produced/composed frame | Production rendering replay matches frame identities, visibility/ownership and independent pixel witnesses under the pinned control |
| Same observed game responsiveness | Correlated presentation outcomes and external game-window witness, plus calibrated capture overhead and co-resident workload measurements |

A GPU composition draw is earlier than Present acceptance and physical display.
Do not label it scanout or user-visible FPS. A window recording can reveal native
HUD/minimap flashing that a map texture cannot; it must disclose missing/dropped
witness frames. Do not add a competing renderer window or presenter to the game.

## Capture boundary and completeness

Extend the existing DLL-owned input/publication and native-adapter boundaries.
Keep injected code as capture/hook glue. Use explicit versioned scalar fields and
length-bounded arrays; never serialize C++ object layouts, padding, pointers,
callbacks, HDC/HWND addresses or GPU handles. IDs include lifecycle generations.
Configuration/asset lookup must resolve from a pinned manifest, not whatever is
installed when the replay happens to run. Asset bytes remain local.

| Input family | Existing owner to instrument | Required data |
| --- | --- | --- |
| Initialization/assets | Renderer API configuration and pack loaders | Schema/build/pack hashes, resolved controls, viewport, executable/driver capabilities, budgets, initial seeds and clock origin |
| World/visible scene | `ScenePublication`, `CapturedScene`, world-page callback return | Full consumed tile/topology/appearance records, authoritative page validity, ordered projected occurrences, visibility/viewer/map epochs and edits |
| Units/actions/resources | Unit API, `UnitInstances`, copied map animation inputs | Identity/generation, definitions and variants, action cursor/anchor, selection, forget/despawn, animation eligibility and captured environmental time |
| Camera/picking | Camera request, validation/adoption and native navigation API | Requests and causes, exact projection, replace/cancel/barrier events, identities and adoption results; picking observes the matching published frame |
| Tactical/native UI | Tactical capture, native image adapter and lifetime registry | Exact operation arguments, clips, colors, text/stroke coverage, source/destination identities, CPU-write payload changes, native access scopes and outcomes |
| Ambient scheduling | Visual clock, retained composition and cadence boundary | Logical sample ticks/frequency, readiness dependencies, offered/skipped/completed opportunities, policy and lifecycle transitions |
| Presentation/recovery | Existing native presenter and ownership handoff | Candidate/accepted/pending/dropped frame identity, transfer rectangle, result, resize/minimize, native handoff, reset/device failure and recreation |

Paths refer to `Renderer/native/` owners. Audit *all* consumed inputs at each
boundary, including rejected requests and native CPU fallbacks. A field is covered
only when a mutation test changes the replay result or is rejected. Missing
families invalidate input-replay acceptance even if the file lasts ten minutes.

Record native CPU-owned image data before its next consuming operation, using the
existing ownership barrier. An escaped pointer can change bytes without a draw
hook; observing function names alone misses it. Immutable/repeated image payloads
should use verified content references and changed rectangles. Actual native UI
pixel inputs are legitimate; renderer-produced map, resource and unit pixels
belong in a separate **oracle** channel, not the input channel for those producers.

Callbacks from replay provide recorded owned values. Workers must never read the
live game's Map/Tile/Unit pointers. Retained-composition recipes must be rebuilt by
the same production owners, not interpreted by a second rendering implementation.
No entire-game emulator, gameplay simulation, API migration, lower quality or
64-bit renderer move is required by this recording contract.

## Three replay modes

1. **Forensic:** pinned build/assets/configuration, original logical visual times
   and causal order. Inspect every output and mismatch, including intended native
   fallback. Worker readiness/adoption outcomes needed for reproducing a race are
   explicit controls. This mode is a correctness experiment, not a speed result.
2. **Performance:** feed the same external input workload and offered visual times,
   preserving causal dependencies. Run real compilers, caches, workers, memory
   allocation, composition and presentation. Measure the candidate's completions;
   do not sleep for the old renderer's work time or force its recorded allocation
   failures. Distinguish external think time from time spent inside renderer calls.
   Faster execution may coalesce/present differently; compare compatible semantic
   frame identities rather than pretending raw frame numbers must be identical.
   The [visible comparison player](realtime_replay_comparison.md) now feeds recorded
   external calls at their arrival times while the production cadence generates
   independent ambient frames. It reports lateness and accepted presentation
   times; recorded native consumption points still constrain this experiment.
3. **Pressure:** repeat the workload under declared capacity/CPU/GPU constraints
   derived from the live envelope. A reserved GiB is a capacity probe, not a replay
   of Civ III heap fragmentation. Report where game scheduling, driver behavior
   and VM contention remain unmodeled.

One run cannot pin the old scheduling outcomes and simultaneously prove the new
scheduler is faster. Establish the forensic oracle first; use repeated performance
runs with the recorder, decoding and oracle readbacks excluded from timed spans
where possible and their remaining overhead explicitly measured. Never subtract
an assumed constant recording overhead from reported gameplay FPS.

## Ten-minute storage and delivery

The older composition writer flushes each event synchronously, captures external GPU outputs
through readbacks, and stops at 512 MiB/180 seconds. The inspected live prefix
contains 471,209 events in 80.788 seconds and 510.745 MiB. Upload/checkpoint/external
pixel records account for about 93.5% of its event storage; submit records only
0.61%. Its linear ten-minute projection is **3.70 GiB**, including startup delays,
not a capacity promise. Peak one-second event storage is about 17.32 MiB. Raising
the old limits alone would preserve the wrong workload and synchronous overhead.

The replacement format needs ordered checksummed segments, 64-bit byte offsets,
a session manifest, per-second/frame index, explicit termination/completeness and
separate input/oracle payloads. Persist exact changed inputs once; retain shared
payloads across logical events. Use a bounded background writer/compressor that
owns only immutable buffers and never calls the renderer, game or GPU. Account
for queued, encoding, in-flight, indexing and oracle staging bytes together in the
32-bit process. Select queue/segment limits from measured burst rates and memory
headroom rather than an unbounded queue. Disk-space preflight and any proposed
external recorder helper are explicit parts of that implementation decision.

If the queue cannot accept an event, stop capture with a precise incomplete reason;
do not silently drop it, reorder it, block gameplay indefinitely or call the
session complete. Commit segments/index entries so a killed game leaves a
recoverable verified prefix. Enqueue time records the producer timestamp, not disk
completion time. Measure per-call capture overhead and queue high-water marks.
Ambient visuals must keep running while native gameplay is busy.

Seek initially replays the complete prefix to rebuild valid state. Fast seek may
later use checkpoints only after they reproduce startup-to-target results,
including native canvas history and ambient recipes. An event byte offset alone
is not a renderable checkpoint. Generate detailed lossless frame witnesses on
demand for a selected range; do not require hundreds of GiB of decoded BMPs for
every ten-minute capture.

## Qualification before another manual capture

The roadmap records current status and ordering. Its required exits are:

- A coverage ledger accounts for every consumed input family above. Intentionally
  missing/altered configuration, assets, visibility, action, native CPU writes,
  clock or reset events cause a precise failure; no external map/pose substitution
  may hide missing producer input.
- Short production captures replay with exact compatible frame/ownership oracles:
  stationary ambient, continuous movement, interturn-like blocked UI, arbitrary
  camera changes, selection/path, reveal/conceal, native HUD/popup, CPU fallback,
  config-off and reset. The old animation-loss sequence fails before its repair.
- A generated **real-time 600-second** workload runs the actual recorder and replay
  at 2240×1260, Standard world size, mixed units and all water effects on. Native
  churn, original/fixed failure cases, memory-pressure intervals, disk-full/slow
  writer, killed-process prefix, corruption and missing-segment controls execute.
  A synthetic timestamp jump to 600 seconds tests indexing only, not duration.
- Capture-on/off paired runs quantify added request/frame tails, address-space
  headroom, allocations, queue occupancy, disk rate and image coverage. Set the
  allowed overhead from the measured envelope before inviting a real ten-minute
  session; do not certify it just because it did not crash.
- Replaying the unchanged control twice reproduces the same semantic outputs.
  Sequential replay versus seek-to-target agrees. Performance mode reports
  request-to-coherent-frame distributions and missed deadlines per trigger, with
  preparation, CPU submissions/waits, GPU work where measurable, composition,
  present/backpressure and whole-process memory separated.

## Systematic optimization decision

After qualification, capture once and freeze that local corpus. Classify slow or
missing frames by trigger and state, not just average FPS: stationary animation,
unit action, input/navigation, scene edits, native handoff, residency pressure and
recovery. Include full-resolution visible/fog/unseen proportions and unit/object
density. Standard (5,000 tiles) is the <33 ms p95 target; Huge (12,800 tiles) is a
separate capacity result. No lowering effects, detail or authored motion speed.

For each architectural hypothesis, state the expected removed work and an
independent rejection test. Change one cause, replay unchanged inputs, compare
correctness and complete latency/memory distributions, and keep or revert the
change. Do not optimize a cheap isolated phase while cost moves into native
adoption or presentation. First resolve whether pressure/eviction, repeated native
composition, producer work, GPU pixel cost, or serialization dominates. Let that
evidence choose the next M4 change, including whether a process boundary is worth
its tradeoffs; do not presume a packet rewrite or bigger budget is the answer.
