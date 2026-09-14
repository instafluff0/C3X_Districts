# Retained world → view → submission implementation

## Current implementation: bounded concurrent preparation

The default full-detail path now has **two CPU helpers**, plus the existing render
thread and single GPU owner. Helpers compile ground, surface details, relief and
vegetation floors from resident shared assets and captured world inputs. Each has
private river/query/layout scratch. Results carry world, coast and river proofs;
adoption validates them before entering the existing compiled-world owner and
selected pass submission. The render thread compiles unstarted demanded work
itself and joins already-running work, avoiding duplicate builders.

CPU preparation is bounded to the captured render/prefetch inputs (at most 8192
jobs), a 16 MiB completed-content reservoir and two river pages per worker. Active
results stop at an 8 MiB buffer threshold; capacity growth can cross it before the
next cancellation check. Byte-based backpressure leaves workers useful while the
GPU owner consumes earlier content. Newly selected demand gets priority over
unneeded speculative results; valid ready demand is protected, and stale CPU
results are rejected before they can suppress replacement jobs. Scratch, active results, thread stacks and
existing GPU owners are additional memory, not part of the reservoir claim.
Configuration, world mutation, reset and benchmark reset join CPU readers before
changing their borrowed inputs. No game pointer or D3D context reaches a helper. Preparation selection and view
assembly share one complete per-frame validity receipt, keyed by geometry epoch
and occurrence anchor. GPU residency is still checked on every use; the 24-byte
receipt is charged to the compiled-content budget. Only stable eligible views
acquire an isolated front for speculative GPU work; moving-camera synchronous
results keep their existing borrowed-output lifetime.

After successive authoritative requests confirm a stationary view, the GPU owner
also prepares at most two future quantized resource-animation frames. Each uses
the existing full-detail scene/pass/finishing path and owns immutable pixels.
Consumption requires exact camera, ordered appearance/topology, native identity,
environment and time-bucket agreement. Foreground changes cancel the horizon;
camera movement does not start it. Snapshot plus speculative publications share
a 32 MiB cap. There is no wall-clock game loop, game-state prediction or redraw
callback. Caller consumption alone advances the finite horizon.

`C3X_RENDERER_CPU_PREPARATION=0|1|2|4` selects the control or helper count; unset
means two. `C3X_RENDERER_PREPARE_AHEAD=0` disables future frames; unset enables
them only for the eligible retained city profile with waves/reflections disabled.
The current custom game configuration already selects that profile. Full terrain
density, meshes, shaders, native overlays, visibility and picking remain intact.

This is scene preparation, unrelated to Civ III worker units. Cities, tree bodies,
infrastructure and units retain their existing content/pose owners; they have not
all acquired new parallel compilers. General caller-driven native asynchronous
camera handoff is still an integration responsibility. A camera miss still waits
for its requested result. No injected changes or new patch symbols are needed.

Preserved control/source and all new evidence are under
`Renderer/native/build/ahead-preparation-20260913/`; the original staged full-detail
DLL is `starting.dll`. The first shallow queue made cold rendering slower even
with four helpers. Replacing tile-count backpressure with byte-based admission
and allowing the foreground to take unstarted tasks is a materially different
scheduler: matched corrected runs reduced the initial dense request from
12.170 s to 7.567 s with two helpers, versus 7.730 s with four. Dense scrolling
remained about 90–100 ms. Four helpers offered no useful gain on the four-core VM.
The corrected scaling run retained more than 1.5 GB of contiguous address headroom. No GPU timestamp
claim is made; the known Parallels timestamps remain unreliable.

The shallow queue and the first combined `ahead-final-*` comparison are retained
as rejected controls. The combined path initially slowed visible edits because
speculative content blocked new demand. Priority admission corrected that. A
bounded endpoint diagnostic then attributed about 5.7 ms of moving-camera cost to
an isolated output copy performed even when no future rendering was scheduled.
Stable-view ownership removed that cost. Per-frame validity receipts avoid checking
the same content twice, but neither correction establishes a scrolling win.

### Final matched work-ahead comparison

`ahead-ownership-build` is the source-matched benchmark build. Runs named
`ahead-ownership-{dense,idle,edit}-{control,candidate}` use identical binaries,
assets, definitions and authoritative requests. The control explicitly disables
both new strategies; the candidate enables two CPU helpers and exact work ahead.
The city profile uses full detail, waves/reflections off, at 2240 × 1192 pixels.

| Complete result | Control | Two helpers + work ahead | Interpretation |
| --- | ---: | ---: | --- |
| Dense initial view, assets loaded and scene reset; two cases | 11.777, 11.933 s | 7.141, 7.048 s | About 40% less latency; serial surface compilation overlaps across workers. |
| Dense scroll, 28 requests; median / p95 | 87.822 / 154.080 ms | 94.997 / 165.754 ms | Slower; mean 97.095 → 100.434 ms. No scrolling-speedup claim. |
| Two visible topology edits; mean | 1372.933 ms | 746.723 ms | About 46% less latency; each checked against an independent cold render. |
| Two distant topology edits; mean | 52.346 ms | 56.355 ms | Slightly slower; no local-edit gain claimed for this case. |
| Stationary resource animation, 30 requests at 67 ms cadence; median / p95 | 34.302 / 39.800 ms | 0.590 / 0.854 ms | Preparation moved ahead of demand; not a throughput gain. |

A same-compiler idle control with helpers enabled and future frames disabled took
31.786 ms median; all 30 images match the prepared publications exactly. The
candidate actually performed 40 future renders, cancelled one and consumed 39,
including warmup: 1532.076 ms of traced work, median 36.707 ms per preparation.
Two consumptions joined active work. Future frames built/uploaded no geometry.
The 67 ms schedule uses absolute deadlines; saved bitmaps can delay demand. In
the same-compiler pair, median deadline-to-completion was 45.082 → 8.567 ms,
including those delays. This is standalone scheduling evidence, not native FPS.
The earlier unpaced witness also passed exact parity, but bitmap writes outside
API calls provide preparation time; its lower waits cannot be called higher FPS.

CPU helpers preserve vertex counts, detail, pass selection and GPU work. They
remove compilation from the serial consumer when a proven result is available;
the first dense construction adopted 813 helper results. Jobs already running are
joined, and demanded unstarted jobs can be compiled by the consumer. The final
repeated dense run records 2815 adoptions including bootstrap/cold cases and
scrolling. These are reuse counts, not additional speedup evidence. The GPU owner
still performs requested rendering and readback. Dense scrolling remains dominated
by completion waits (roughly 44 ms median in the control); more CPU workers cannot
remove that cost. Do not expand helper count or another output-helper experiment
on the strength of cache-hit counts.

The maximum completed CPU reservoir in this pair was 12,722,492 bytes of its
16 MiB cap; exact future-frame ownership peaked at 24,278,372 bytes of its 32 MiB
cap. The final dense candidate retained at least 1,520,304,128 bytes (1.42 GiB)
of contiguous free address space. Across the eight final workloads the minimum
was 1,368,051,712 bytes (1.27 GiB), above the 512 MiB floor. Private scratch, active
builds and thread stacks are additional to the preparation caps. Four helpers
remain an explicit measured option, not the default: their corrected cold result
was slightly slower than two and used more aggregate compiler time.

The CPU compiler changes 13 of 2,670,080 pixels by one channel level in the dense
control comparison. This is a new recorded precision difference, not an approved
visual change or a detail cut. Worker counts agree with one another; full-detail
controls, assets and fixed references are preserved.

The full checkpoint before the final scheduler/ownership corrections passed 268
tests (one existing skip), scrolling, reduced zoom, wrapping, resource
animation/scroll/removal and day/night unit bodies/lifecycle/terrain parity.
The old unit witness incorrectly expected identity-driven idle phases; it also
failed against `starting.dll`. It now supplies distinct native cursors and checks
that identity/time cannot change a fixed cursor. Production unit behavior is
unchanged. Current corrections pass 48 focused executable tests; final production
integration verification and evaluation staging are recorded below when completed.
No installation, game launch, Git mutation or native asynchronous camera integration
is included in this work.

## Current implementation: shared geometry and terrain patches

The user selected **full detail** for Civ III evaluation. The verified staged DLL
uses the original terrain density (`C3X_RENDERER_PATCH_PIXELS=0` by default), with
shared connectivity and tree instancing enabled in the eligible retained profile.
Reduced-detail measurements remain preserved diagnostics, not the delivery mode.
There is no temporary low-detail image or automatic refinement in this build.

The existing CPU-phase diagnostic places the largest geometry cost in terrain:
2068 ms in natural ground, 3203 ms in relief, and only 40 ms in tree expansion on
the preserved dense initial scene. Therefore a forest-only instancing claim would
miss the dominant cost. The connected implementation has two responsibilities:

- Terrain patches separate shared canonical connectivity from locally valid
  surface values. CPU layouts and GPU index buffers are reused by all compatible
  patches; color and shadow passes consume the same content. An explicit
  screen-space detail policy selects a common power-of-two mountain lattice for
  the view and its collar neighbors. Zero preserves existing density; positive
  settings are visual candidates requiring review, not equal-quality speedups.
  The pixel setting describes nominal projected horizontal grid pitch, not a
  proven image-error bound on sloped relief.
- Immutable tree placements reference one resident body mesh per generic asset.
  The existing world owners retain placement and exclusion/height dependencies;
  view assembly supplies authoritative projection; selected color inputs form
  ordered instance batches, and shadow casters consume the same placement data.
  Source assets, density, placement, clipping and materials remain unchanged.

The existing mesh budgets remain in force. Shared terrain indices are capped at
4 MiB and charged in geometry admission; instance source meshes are capped at
32 MiB; each pass stream is capped at 1 MiB. Compatible batches append into unused
stream storage and discard only at wrap, avoiding a full stream rename per batch.
Device/reset/eviction and publication continue through existing owners. Reflection/legacy profiles retain their existing
geometry execution. Native asynchronous handoff remains outside this work.

Preserve the current uncommitted source and verified binary control; test shared
layout/placement ownership and both GPU passes, then compare stationary, dense
scrolling and local edits through complete checked results. Distinguish unchanged
quality effects from detail-policy tradeoffs. No Git mutation, installation or game
launch. Evidence: `Renderer/native/build/mesh-instances-20260913/`.

Current verification passes: 259 production integration tests (one skipped),
resource animation/scroll/removal and common-depth checks, 21 focused ownership
checks, and final boundary/zoom/content replays. The latter include four topology
changes and six city/forest appearance changes checked against independent cold
renders. Sources, binaries and runtime inputs remain unchanged within each run.
A stale extracted test fixture now includes the shared instance type. The initial
suite attempt used an older Python without Pillow and lacked VM access; its errors
are preserved and superseded by the successful bundled-runtime integration run.

A first complete comparison did not establish a scrolling win. Its initial
per-batch stream discarded 1 MiB for every small submission; the connected
correction uses bounded append/no-overwrite storage and discards on wrap. The
D3D content replay now submits 50 color and 167 shadow batches with one
initial discard per stream, then none on the following request. Earlier runs,
shader-cache warmups, transport failures and the one dense run overlapping CPU
tests remain preserved; final acceptance timings run without builds or tests.


### Final shared-mesh comparison and delivery

The preserved starting source/binary and final `shared-instances-stream-build`
use the same art, definitions, world data and request sequences. Sixteen enumerated
implementation inputs differ: the new instance shader and its generated adapters,
compiled shader caches/provenance, and the shader-cache ignore file. Every run
verifies its own unchanged inputs, source closure and binaries. The normal city
profile has waves/reflections off. Dense scrolling uses two 14-request sequences;
stationary animation uses 30 requests after ten warmups; edit means contain two
distant and two visible changes, each compared with an independent cold redraw.
Initial views start with assets loaded and renderer caches cleared.

| Whole request | Starting control | Full detail | 3-pixel detail candidate |
| --- | ---: | ---: | ---: |
| Initial dense view | 13033.80 ms | 12354.74 ms | 8611.38 ms |
| Dense scrolling | 99.05 ms | 99.76 ms | 95.35 ms |
| Stationary animation | 27.56 ms | 26.32 ms | 25.23 ms |
| Distant topology change | 70.95 ms | 68.11 ms | 59.17 ms |
| Visible topology change | 1555.52 ms | 1403.20 ms | 945.94 ms |

Full detail improves initial construction by 5.2% and visible edits by 9.8%; dense
scrolling is effectively unchanged. The detail candidate improves initial work by
33.9%, visible edits by 39.2% and scrolling by 3.7%. Stationary static work was
already skipped; its timing variation does not establish an instancing benefit.
These are standalone checked-result latencies, not native presented FPS.

Initial geometry uploads fall from 317762132 to 278106174 bytes at full detail,
or 197937814 bytes with the detail option. New shared asset meshes total 112616
bytes; initial color/shadow instance streams add 179904/264256 bytes, reported
separately from geometry uploads. Dense requests still build 26.86 entries and
upload 421114 geometry bytes, adding about 7090/20622 instance bytes. They mostly
reuse terrain already constructed. Shadow source draws fall from 284.86 to
170.43 per request without changing the 3.14 rebuilt pages. The append correction
eliminates discards during the measured dense playback. Smaller uploads and fewer
submissions have not eliminated its approximately 50 ms completion wait.
Parallels GPU timestamps remain unsuitable for attributing that wait to a specific
GPU stage. No new output-helper campaign or native handoff was substituted.

Renderer target accounting stays at 1165363200 bytes, distinct from actual process
residency. Shared mesh/index/stream caps stay bounded. The minimum reported
contiguous free region across full-detail workloads is 1351.74 MiB; the detail
candidate's minimum is 1432.34 MiB. Existing cache and target budgets were not enlarged.

Across all final saved images, full detail differs from the starting renderer in
at most 23 of 2670080 pixels, with maximum channel difference 5/255. This is new
numerical variation in equivalent geometry/shader calculations, not byte-exact parity or an
already accepted visual change. The detail option changes up to 9.8% of pixels,
including relief silhouettes and shading; it is not an equal-quality speedup.
Both preserve same-mode cold-render parity and returned ownership checks. Review
`mesh-instances-20260913/final-terrain-comparison.png` and
`final-terrain-detail-crop.png` beneath the native build evidence directory.
No fixed references were replaced; `C3X_RENDERER_PATCH_PIXELS=0` remains default.
`--patch-pixels 3` selects the tested detail candidate in the existing harness;
`--legacy-tree-meshes` isolates the old tree geometry mechanism when needed.

The verified production DLL is staged **for evaluation**, under the earlier user
authorization, at `Renderer/bin/C3XRenderer.dll`, SHA-256
`ae7a7306adbe756af8c4b9cd73375ea242b616570ec95bd423d6b8143e22b70e`.
Its passing production receipt, prior DLL and exact replacement are preserved in
`mesh-instances-20260913/evaluation-staging/`. No installation, game launch or Git
mutation was performed. Native caller-driven asynchronous handoff and live cadence
remain explicit integration responsibilities. Reflection/legacy execution and other
object categories retain their existing paths. The user selected full detail; the
reduced terrain option remains disabled for delivery.

Final comparison data/scripts and the starting source snapshot are preserved in
`Renderer/native/build/mesh-instances-20260913/`. Final candidate runs are
`Renderer/native/build/shared-instances-final-{dense,stationary,edit}-{full,detail3}`;
control runs live in that snapshot. Rehydrate `control-snapshot` at
`Renderer/validation/mesh_instance_control` before reproduction, because runtime
input discovery excludes directories beneath `build`. The new shared path replaces
baked trees in its eligible profile; there is no separate renderer framework.


### Remaining native integration recommendation

Complete the native caller-driven asynchronous preparation/publication handoff,
including the displayed-view contract described in
[native asynchronous presentation](native_async_presentation_audit.md). The retained
world and selected passes supply full-detail work to the existing bounded worker
queue. Separate requested state from the actually displayed view so pixels, native
overlays, visibility and inverse picking always agree. A later Civ III rendering
call may adopt a compatible complete publication; completion must never notify
Civ III or request a redraw. Pending input should coalesce without starving useful
full-detail completions. This requires more than binding the existing begin/poll
exports: the no-ready-image case must be correct before enabling async camera mode.

Validate stationary animation, sustained navigation, zoom, visibility loss, local
edits and unit takeover through that complete integration. Measure native caller
blocking, input-to-correct-presentation latency, displayed-frame age and complete
render-job time separately. This targets responsiveness and finishes the remaining
end-to-end ownership responsibility; it does not claim to remove the roughly 50 ms
pending-GPU-work wait or make a full-detail job intrinsically faster. Preserve the
current full-detail control, closed resolve findings and GPU-timestamp limitations.
This is a recommendation, not a new native implementation started in this handoff.

## Objective and control

The user's connected architecture instruction supersedes the former experiment
queue. Complete ownership and behavior through the existing renderer; GPU mesh
residency remains bounded. Preserve native ownership, asset/visual contracts and
deferred scope. Native asynchronous handoff remains a separate integration gap.

Incremental output is finished. At the start, candidate source verification passed and
the staged DLL matched its receipt (`b5cd1b08…c1f4d`). Its final dense comparison
was 119.40 → 103.28 ms; stationary work showed no improvement. Exact finishing,
edit and boundary witnesses passed. Evidence, rejected resolve approaches and
staging are preserved in [the output record](history/retained_output_completed_20260913.md).
Full hardware resolve and output handling are frozen for this implementation.
Parallels GPU timestamps and event-query completion are unreliable attribution.

## Six responsibilities: code assessment

Completion below concerns ownership and behavior for the admitted map plane, not
permanent GPU residency, an uncaptured game simulation, or every optional backend.

| Goal | Starting gap | Implemented responsibility and limits |
| --- | --- | --- |
| Persistent world/instances and reusable content | The 8192-entry appearance cache forgot identities; world records also held camera observations. | `CapturedScene` now retains canonical render appearance/revisions/bindings separately from a reusable observation pool. Camera departure and GPU eviction preserve identity. `ResidentContent` and existing mesh owners remain the only GPU lifetime authority. Never-captured appearance stays unknown. Metadata admission is bounded and fails explicitly. |
| Local revisions and complete content validity | Appearance checking was split; outer hits bypassed forest/city exclusion validity; river queries did not propagate complete dependencies. | The instance fast path and ordinary cache lookup share validity. Own appearance, neighboring city presence/appearance, semantic neighbors, coast nodes, world samples, local river nodes and river query cells govern reuse. All three compiled tiers carry river proofs. Native population/labels/selectors and unit state do not invalidate static content. Broad natural/ground topology keys are replaced by these proofs. Asset/device/world-basis contexts still invalidate globally. |
| Spatial selection separate from world construction | Compilation appended active draw lists; circular spans scanned every layer. | Compilation publishes protected handles. A separate assembly consumes current authoritative occurrences; the view index selects actual static pass inputs for damage spans, with exact intersection and ordered deduplication. Current capture remains the admission set. The index is view-scoped; world identity and compiled validity outlive it. Dynamic pose lists retain their small scan. |
| Compatible submissions through explicit passes | Ordered layers/page limits existed, but selected inputs were incomplete. | Selected static/dynamic inputs feed existing ordered material submissions and bounded 32-page receiver batches. Main/caster/resource ownership and common depth are preserved. This completes representative pass ownership; selected tree placements now feed hardware-instanced color and shadow submissions. Reflection and other object categories retain their existing execution. |
| GPU reuse, finishing and readback | Incremental output already finished. | Complete for the tested city profile, waves/reflections off and bounded extents. Circular color/depth, sparse restore, incremental finishing, hardware resolve and consolidated readback remain unchanged. Other profiles retain existing output execution. |
| Caller-driven asynchronous preparation/publication | DLL preparation/publication and exact queued joins existed. | Bounded CPU preparation and exact future ambient publications now run ahead of demand inside the DLL. **Remaining native integration responsibility.** General asynchronous native camera handoff and presented cadence are not implemented. Unmatched current-camera demand still waits for exact output. No renderer notifications, redraw requests, new native hooks or ABI changes. |

## Ownership and invalidation details

The world topology retains authoritative whole-map terrain. Canonical appearance
records retain only render-relevant state observed through the capture contract;
full records take precedence over duplicate lightweight halos. The observation
pool is capped at 8192 occurrences. Persistent record admission and conservative
metadata accounting are capped at 128 MiB, rather than evicting logical identity
on camera pressure. Explicit renderer/world reset restarts that ownership.
Compiled meshes remain evictable under the existing geometry/CPU budgets.

`NaturalWorld` now owns a dependency chain from copied world inputs through river
pages to exact ordered query-cell values and compiled consumers. Missing cells
are observations too. Source proofs survive page eviction, so validating unchanged
meshes does not reconstruct fields. On an actual source change, the exact queried
cell values determine consumer validity. Page construction still uses the original
expressions and the globally invalidated 16-page cache. This differs from the
closed page-cache experiment: it establishes **compiled mesh** validity, not merely
page reuse. Source inputs are capped at 4096/page, cell metadata at 512 KiB/page,
and compiled observation sets at 256 cells. Retained proof payloads are charged
conservatively to existing compiled-cache budgets, including shared data.

The first broad proof propagated every page input directly to every consumer.
It passed redraw parity but increased the small visible-edit cost to about 3.1 s.
Exact cell proofs restored the same workload to about 1.29 s. That intermediate
source and failed control setup receipts remain in the evidence directory.

## Connected design

1. Separate persistent canonical appearance/identity from current observations.
   Retain identities across camera departure and GPU eviction; bounded admission
   fails explicitly instead of silently forgetting the world. Only current full
   captures authorize appearance, selection or replacement.
2. Carry complete appearance and actual observed dependencies through every
   compiled tier. Forest exclusions observe city appearance, including absence;
   river fields connect construction and height-callback inputs to exact query-cell
   proofs retained by consumers.
   Remove broad mesh keys only with these proofs. Preserve the globally invalidated
   small river-page cache: this completes compiled-content validity rather than
   reopening the rejected page-cache experiment.
3. Compilation publishes protected content handles. Separate view assembly consumes
   current authoritative occurrences. Reuse the contributor index to select actual
   ordered pass inputs for circular damage, rather than only raster keys. Preserve
   exact intersection, layer/occurrence order, bindings and shadow-page limits,
   with bounded fallback scans.
4. Preserve starting source/build and a same-binary control. Verify portable
   ownership/validity/selection contracts and independent replay correctness, then
   whole-request stationary, dense-scroll and local-edit comparisons. Report
   eliminated work and initial construction separately. Automated tests do not
   imply staging, installation, launch or visual acceptance.

## Previous world/view checkpoints

Implementation: independent topology edit and six city/forest appearance edit
checks pass; the latter hold topology revision fixed and exercise growth, removal,
style changes and reversals. All preserve exact pixels, replacement flags and
animation ownership against cold redraws.

Final-source portable ownership, dependency, selection and measurement checks pass
(34 tests). The shared river witness preserves all 312 original-expression samples,
wrapped lookup, revision invalidation and the 16-page LRU. Native final-source
replays pass six retained-boundary cases, zoom and ambient transitions, four
topology edits and all six fixed-topology appearance edits. Each replay receipt
confirms unchanged sources, binaries and runtime inputs. No new visual difference
is expected or requested for approval; earlier accepted visual contracts remain.

Production verification passes via `python3 Renderer/renderer.py integration
resources --renderer-only`: 257 tests run, one skipped, plus config-off/playback,
scroll/removal parity and native-depth checks. Three extracted C++ fixtures were
updated to include the new dependency owner and preserve the correct extraction
boundary; the ground-cache witness now checks both local validity and the old
global-revision diagnostic control. Parallels transport failures are preserved
in the earlier logs and superseded by the successful final run. No injected code
changed, so injected compilation was not required.

Under the earlier authorization to add the renderer for Civ III evaluation, the
verified production candidate is staged at `Renderer/bin/C3XRenderer.dll`, SHA-256
`22e87ecc770005ab3aa84074ab6877072b2b0755352301cb21a7a846370ed96b`.
The previous DLL, new DLL, source/build receipt and passing integration receipt
are preserved in `world-view-20260913/evaluation-staging`. No installation or Civ III
launch was performed. Native live acceptance remains unmeasured.

### Previous world/view whole-request comparison

The preserved starting binary (`incremental-output-delivery-build`) and final
`world-view-validated-build` use identical runtime dependency hashes and matched
normal-tier city profiles, waves/reflections off. Dense scrolling comprises two
14-request runs; stationary animation measures 30 requests after 10 warmups. Local
edits contain two distant and two visible changes, each checked against a separate
cold redraw. Initial construction is excluded and reported separately. Edit runs
use a 180-second limit; the earlier timed-out control is retained and excluded.

| Workload | Control | Current | Observed change |
| --- | ---: | ---: | ---: |
| Dense scrolling | 109.30 ms | 103.96 ms | 4.9% faster |
| Stationary animation | 28.14 ms | 26.77 ms | 4.9% lower; no new eliminated rendering work |
| Distant topology change | 65.07 ms | 63.96 ms | 1.7% faster |
| Visible topology change | 1459.74 ms | 1495.37 ms | 2.4% slower |

Dense preparation falls from 36.18 to 31.62 ms, with 1030.29 ready instances per
request. Both arms build 26.86 entries and upload 421114 bytes per request; resolve,
finishing and readback work are unchanged. A separate same-binary attribution pair
(`ready-dense-control` / `ready-dense`) measured 120.03 → 114.73 ms and the same
build/upload counts: the old submission scan inspected 13979.71 candidates versus
921.93 indexed static candidates plus 507.43 dynamic candidates. Selected inputs
and batches remained identical. That pair isolates the new mechanism; it is not
the preserved-source headline control.

Stationary timing varies despite identical output work, so do not attribute its
entire observed improvement to this architecture. GPU completion remains about
50.69 ms of the current dense request, versus 52.60 ms in the control. Those are
CPU-observed waits, not reliable GPU-stage timestamps. Initial dense preparation
still takes 12.90–13.73 seconds (control 12.59–12.86). Visible edits still compile
81 entries, reuse 976, and spend 1247.63 ms in geometry (control 1195.81). The local
validity correction does not make the existing terrain compiler cheap. This is a
modest scrolling improvement, not a large universal speedup or a native FPS claim.

All common saved comparison images are exact: 5 dense, 32 stationary and 3 edit
images. Renderer-target allocation accounting is unchanged at 1165363200 bytes;
this is not a process-residency measurement. Dense world metadata averages
2.40 MiB. The smallest reported contiguous free region is 1532 MiB for candidate
dense scrolling and 1443 MiB for its local edits. Existing geometry/output caps and
the new metadata/proof caps continue to bound ownership in the 32-bit process.

### Evidence and reproduction

`Renderer/native/build/world-view-20260913/` preserves the starting commit/source,
intermediate broad-proof implementation, comparison script/JSON, logs, and rejected
setup receipts. Complete runs are `Renderer/native/build/world-view-final-{dense,
stationary,edit}-{control,candidate}/`; each contains input/binary/source receipts,
request endpoints, ownership checks and trace data. `C3X_RENDERER_RETAINED_WORLD=0`
remains a same-binary diagnostic control; it is not the historical source binary.

The isolated original-source snapshot is preserved under the evidence directory
as `control-snapshot`. To rerun it, copy it to
`Renderer/validation/control_snapshot_20260913` because runtime discovery excludes
paths beneath `build`. Its local runtime inputs are verified hard links/copies,
not redistributable assets. Its measurement harness also hashes the 111 textures
actually referenced by the city material table outside the declared pack folders;
the original dependency inventory omitted those files. Both final arms include
the same complete 10315-file runtime set. No assets were discarded.

### Remaining implementation responsibility

Native caller-driven asynchronous handoff remains explicit: capture/prepare on
Civ III calls, publish only matching completed output on subsequent calls, and
retain exact visibility, overlay, picking and config-off behavior. No unsolicited
renderer callback or redraw mechanism is appropriate. Shared tree instancing now
completes a representative asset/placement/pass path, while broader category and
reflection instancing remains outside the measured profile.
Future performance work should address the measured geometry construction and GPU
completion costs; this change does not reopen output-helper experiments or expand
native ownership, wonders, or Districts scope.
