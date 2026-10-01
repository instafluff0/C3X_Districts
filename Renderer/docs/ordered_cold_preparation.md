# Ordered publication and cold preparation

Renderer64 separates native command receipt from helper execution and camera
adoption. The asynchronous bridge copies each accepted input, reserves local
image identities, and resolves them against the executed prefix. Consecutive
image requests can share a transport batch. Create, upload, draw, saved-version
and destroy order is preserved; camera, unit, tactical and presentation requests
fence a batch. Negative batch-local identities refer only to earlier creates in
that batch. Failed execution returns a prefix and faults the publication owner;
the unconfirmed suffix is never acknowledged as completed.

Publication retains the existing 128 MiB / 8,192-record limits and additionally
charges at most 8,192 semantic work units. Each image operation charges one unit,
each draw command another, and uploads charge by 256 KiB blocks. Joining a batch
does not reduce any charge. A joined batch contains at most 64 image operations,
4,096 work units and 1 MiB; a single larger upload still uses the existing 16 MiB
wire limit. The helper admits one bounded batch, returns a receipt, and services
execution polls while its executor waits for the renderer owner. It does not
own an accumulating second backlog or a second graphics context.

Pending camera snapshots can supersede pending snapshots. Reliable native,
action and lifecycle commands retain their order. A receipt-only cancellation
mailbox reaches the helper independently of an occupied request wire. It marks
an obsolete admitted camera ticket; the immediate-context owner consumes that
mark at preparation checkpoints. Configuration and reset serialize mailbox
access against worker retirement. Identical camera observations differing only
in presentation time/ambient animation count share their pending destination.
Projection, native visibility, topology, content and anchors still participate
in equality. Camera receipt, non-adopting readiness and ordered adoption remain
separate operations.

Cold compilation runs through the existing immutable CPU preparation workers.
Waiting for a ready chunk releases the preparation mutex before calling the
renderer service checkpoint. On the renderer thread, a checkpoint can execute
one independent native image/tactical/presentation command through the existing
dispatch. Completed chunks, generation leases and the active render stack
survive that turn. Worker cancellation callbacks do not access the immediate
context. UI updates do not cancel and restart the whole scene. A superseded
camera discards its selected view while valid completed content remains reusable.

The completed old map remains a valid native composition source during cold
preparation. Its ambient callback freezes while mutable destination scratch is
occupied. Continued old-view presentation is therefore recorded separately from
first-correct destination and full-quality destination responsiveness.

Counters distinguish accepted records, fully executed records, safely
superseded camera snapshots, abandoned unconfirmed work and rejected admission.
They also report actual executed adoptions, presentation calls, pending bytes,
records/work, their high-water marks and oldest queue age. Verbose publication
latency includes these counters and separate queue/service intervals; ordinary
quiet cadence runs do not enable it. These are API presentations, not scanout.

Pressure never silently drops a reliable command. It faults the generation and
reports once. Reconciliation joins the failed prefix outside the frame path,
retires scene/resources/native identities through reset, and clears the fault
only after that reset succeeds. Normal teardown can instead abandon the native
owner and retire its helper. Failed reset leaves publication unavailable.

Shadow atlas identity includes every selected draw dependency: immutable content
generation/version, binding/layer, bounds/offset, vertices and indices, vertex
and index offsets, index format/count, stride, immutable instance identity,
instance material and rigid mode. Adversarial shared-vertex tests ensure
different index ranges or instance streams survive deduplication and validation.

## Validation

Private receipts live under `Renderer/.cache/ordered-cold-step/`. Host contracts
execute the actual queue, batch alias resolver, preparation wait and extracted
renderer checkpoint. Native fixtures exercise saved versions, UI aliases,
blocked producer/consumer intervals, camera adoption and independent presentation.
The opt-in helper/preparation delays use bounded elapsed-time deadlines and are
disabled in ordinary rendering. Validation and integrated game qualification
are recorded with the final matching source/binary closure in that evidence root.

The candidate is **unqualified for game staging**. Native 128-pixel camera
handoffs pass, including bounded elapsed-time preparation/helper delays, eight
rapid camera demands, ordered saved versions and a suspended consumer. The final
fixture accounts for 1,359 accepted records as 1,352 executed plus seven safely
superseded snapshots, with zero pending, abandoned or rejected work. Its 32
destination handoffs reach readiness within 860 ms; the longest caller poll is
3.147 ms. Normal-effects native 64-pixel coverage passes at 960×720 with eight
units and 24 real pose changes. These are fixtures, not game cadence.

The 1498 AD game fails before the first scheduled minimap jump. Camera one
completes and is adopted; the first busy retained-display callback then takes
1,120.190 ms, including 589.745 ms of demanded preparation for 76 unit samples
(431 part samples, 76 main and 72 reflected contributors). This callback invokes
FRESH drawing directly rather than the camera-render checkpoint. The last
completed publication is record 2,710. At 30.018 seconds, admission faults with
2,196 pending records, 7,391 work units and 6,003,764 bytes; the incoming request
would exceed the semantic limit. The smaller semantic bound exposes this busy
display failure earlier than the control's known cold-jump packet exhaustion.

The remaining blocker is resumable demanded unit payload/pose preparation and
ordered command service inside the retained display transaction. Servicing image
commands recursively during an unfinished draw would need a safe pass boundary
and retained ownership proof; the camera checkpoint alone does not provide it.
The measured display span also includes 2,238 native UI operations and 286 copies,
so its full duration cannot be assigned to unit loading or HUD replay alone.
`candidate-jump-blocker.json` preserves the raw log hashes and correlated timeline.
A repeat with detailed tracing disabled also faults at 29.897 seconds: 2,185
pending records, 7,358 work units and 5,995,896 bytes, with 4,960 accepted and
2,775 executed records. Both runs complete the first canonical map. This confirms
the qualification failure persists without verbose publication observation;
the detailed run supplies attribution, not a quiet candidate performance result.
Automated input stops on the first fatal interval: only the two load keystrokes
were sent, with no minimap clicks or gameplay saves. Candidate quiet scroll/zoom
comparisons and all six jump transitions remain unqualified. The accepted control
trio is restored; it also retains its previously demonstrated cold-jump failure.

Fresh quiet control measurements are 6.136 presentations/sec while scrolling,
12.162 idle, 9.383 during zoom/reversal and 16.997 after settling. These API
presentation rates do not meet the performance targets and do not establish a
candidate improvement. No valid GPU timestamp attribution is claimed.

The selected contract set passes after serial native retries. Five inherited
source assertions remain excluded with their earlier baseline receipt. A separate
terrain-pool parity assertion also fails with the unchanged foundation header;
its preserved baseline reproduction is `pool-baseline-contract.json`, and it is
not claimed as fixed. The final completion receipt records test scope, source
and binary hashes, runtime inputs, preserved evidence and exact cleanup.

No injected hook, patch-table signature or rendering ownership changes are
required. The previously documented full-resolution native-64 coastal-wave
capacity limit remains separate unless explicitly closed by a matching receipt.
No reference image is replaced by this implementation.

The larger color difference inherited from the persistent-scene foundation is
specifically attributed by the receiver-bound ablation described in
[persistent scene and shared passes](persistent_scene_shared_passes.md).
The shadow identity correction and immutable instance buffers do not explain
that difference. Independent full-frame and depth comparisons remain necessary;
neither small mean color error nor exact depth establishes imperceptibility.
