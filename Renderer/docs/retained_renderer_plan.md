# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content; nearby preparation helps
without making every possible destination image a prerequisite for fast movement.
Full detail, authored animation and native ownership remain unchanged.

**Active deliverable: milestone 1.** Its replacement static submission path is
implemented and validated, including packed terrain preparation; its whole-request
performance criterion remains open.
Continue the missing-content responsibility identified below without another
planning/approval gate. Milestones are connected outcomes, not promised short
passes. [Architecture](renderer_architecture.md) owns the design;
[validation](benchmark_workflow.md) owns measurement and acceptance.

## Current implementation and gaps

| Original responsibility | Implemented foundation | Remaining responsibility |
| --- | --- | --- |
| Persistent world/instances | Camera-independent content, shared assets/forest meshes, revisioned units, packed terrain records, immutable mesh ranges/materials, bounded residency | Remove remaining foreground construction of missing content; broaden mesh sharing where useful |
| Local validity | Captured appearance/dependency revisions, unit revision/despawn proofs | Preserve complete validity through migrated representations; refine authoritative change publication where useful |
| Spatial selection | World pass index and native/wrapped occurrences feed replacement static submissions directly | Collected dynamic/unit pass inputs |
| Compatible passes | Compatible static layers, shared material bindings, batched occurrence parameters/uploads, forest instancing, exact native composition | Collected dynamic/unit execution; additional sharing guided by cost |
| GPU reuse/output | Resident admitted map/poses, static color/depth, incremental finishing and native composition | Direct dynamic scene execution; reduce per-pose and full-map work where measured |
| Async integration | Bounded content/pose/view preparation, selected-work urgency and independent visual timer | GPU-ready preparation for remaining objects; coherent general nonblocking camera/content publication |

Independent frames use the existing HWND presenter without native redraw requests.
Civ III's original gameplay timer is unchanged. Map animation still enters the
existing render orchestration; units still produce separately finished poses.
These are working mechanisms, not yet the cheap scene-frame endpoint.

## 1. Complete world → selected GPU submissions

Connect persistent content, shared mesh/material bindings, compact instances,
spatially selected occurrences and explicit compatible pass inputs. Extend current
owners and replace migrated submission paths together. Include worker ownership:
prepare dependency-ready content and selected pass inputs concurrently, separate
current-frame work from speculation, and replace redundant queues where needed.
This is part of milestone 1, with animation scheduling extended in milestone 2;
it is not another milestone. Do not deliver only a command/job abstraction or
instancing that leaves submission costs unchanged.

Start with the active dense-scrolling map path, including terrain, vegetation and
representative repeated city/infrastructure content. Choose actual conversions
from code and bounded attribution; CPU-transformed vertices already cached on warm
requests are not automatically the dominant cost. Preserve unique/deformed meshes
and exact material/depth ordering where sharing or sorting is incompatible.

**Done:** pans reuse unchanged content; local edits affect their real dependency
neighborhood; selected inputs drive the replacement production submissions.
Demonstrate whole-request effects, eliminated construction/binding work and bounded
memory on stationary animation, dense scrolling and local changes.

## 2. Make independent animation a direct scene operation

Sample eligible animation state, collect required resource/unit poses, prepare
missing content together and execute selected dynamic/shadow/finishing passes over
valid static color/depth. Feed results into the existing native compositor. Keep
native overlays, unit ordering and terrain occlusion exact; static UI stays reusable.

Shared pose buffers, GPU deformation/skinning and grouped targets are candidates
where they remove measured work. Do not force all units into one surface or alter
source playback just to simplify batching.

**Done:** idle animation does not rebuild world content or rediscover static
submission state; animated-unit scaling avoids repeated independent setup/finish
work where compatible. Verify frozen units, authored work loops and native actions.

## 3. Complete state publication and nonblocking camera updates

Keep durable world/lifecycle updates separate from replaceable camera requests.
Publish copied authoritative changes and exact view eligibility; refine mutation
hooks only where needed. Prioritize current missing content, then nearby reusable
meshes/instances/pass inputs, then selected future views/animation pixels. Preserve
useful preparation across supersession without evicting the current working set.

**Done:** ordinary scrolling/zoom across useful coverage avoids synchronous
construction barriers; cold, evicted and invalidated destinations have bounded,
correct handling. Pixels, camera, visibility, overlays and picking advance
coherently. Never hide a wait by presenting mismatched or stale view identity.

## 4. Raise cadence against a measured frame budget

Tune sustained presentation only after frame costs and publication are coherent.
Measure input-to-display latency, frame tails, sample age, worker interference and
combined memory. Consider parallel D3D recording or less UI-thread dependence only
if the remaining cost warrants it. Stay on D3D11 unless evidence justifies a change.

**Done:** demonstrate the selected cadence on representative live workloads,
including navigation, animation and UI transitions. A 33 ms timer is not 30 FPS;
60 FPS is a later 16.7 ms frame-budget objective, not a current promise.

## Current evidence and implementation handoff

**Completed:** terrain workers now return packed shader vertices, compact indices,
bounds and existing world/coast/river proofs. The foreground raw-terrain handoff
and duplicate packing are replaced. Shared-grid topology and immutable GPU ranges
remain reusable; other object meshes use the same packer in the foreground.
The render owner finishes independent content before joining active helpers, then
assembles occurrences in native order. No new pool, timer, hooks or quality cuts.

**Measured:** the full-detail 1120×1192 fixture includes map, eight units, native UI
and final transfer, with capture outside timing. Two final candidate runs and two
preserved-control runs each contain 64 GPU samples per workload:

| Workload | Control means (ms) | Candidate means (ms) |
| --- | --- | --- |
| Stationary animation | 8.67, 9.23 | 9.39, 9.37 |
| Dense scrolling | 41.58, 58.64 | 39.22, 42.36 |
| Local change | 10.94, 11.31 | 10.27, 11.09 |

Scrolling preparation averages 19.88–21.53 ms versus 22.89–32.55 ms. A comparable
55-content trace eliminates repeated packing for 108 terrain meshes; helper wait
falls from the first packed candidate's 7.91 ms to 0.05 ms after correction.
Those spans overlap other work. The control's second run slowed sharply mid-run;
**a reliable overall speedup is not established**. Idle/local changes are broadly
unchanged. All connected receipts retain exact control pixels and native ownership.
Sampled contiguous headroom remains at least 1.26 GiB in the final runs.

**Next unfinished responsibility:** remove the remaining foreground ground/object
assembly and adoption cost, using the completed packed-content boundary and actual
cost attribution. Missing-content preparation still consumes about half a scrolling
request; simply moving more work to helpers is insufficient. Preserve shared assets,
local validity and native occurrence order, and keep milestone 1 acceptance open.
Direct dynamic/unit scene execution remains milestone 2; general coherent
nonblocking native camera publication remains 3. Do not start another output-helper
or speculative-coverage detour.

The preserved performance-control DLL is
`a251c38008c6d090be78799419d40b8225a5ae4da2e5b8b5466fc4e2fc413fbd`; its checkpoint
had 289 passing tests and one platform skip. Current staging is recorded below.
No game installation or launch was performed. Exact controls and verification are in
[the packed-content checkpoint](history/gpu_ready_content_20260915.md).
[Submission evidence](history/selected_submissions_20260915.md) and
[earlier findings](history/retained_renderer_checkpoints_20260915.md) remain preserved.

### Current bug-repair checkpoint

Live feedback confirms unit selection and the previous HUD fixes. The new black
rectangles were reproduced: canceled camera work discarded draw inputs, but idle
static-cache preparation cleared regions and certified them with no contributors.
Abandoned views now retire their draw/raster validity together; background work
requires complete inputs. Compiled world assets and published fronts remain owned.
The old build fails the scrolling reproducer at step 2; the fix passes 40 steps
across cancellation, both scroll axes, zoom and idle preparation.

The latest screen-upload log is 16-bit data and describes **2240×1192**, correcting
the earlier 1120×1192 interpretation. Native map residency was lost after startup,
explaining the lack of independent animation frames. At the actual extent,
map/screen/popup pairs exceed the old 64 MiB live-composition budget; the Session
now has a measured 96 MiB ceiling. Other resource budgets and full detail remain
unchanged. The maximum extent has no wider prepared donor: camera demand remains
exact and synchronous there. General nonblocking publication is still unfinished.

[Repair evidence and controls](history/resident_scroll_repair_20260915.md) preserve
the reproduction, earlier unit/sampler fixes, complete delivery-route measurements
and test limitations. At 2240×1192, 384 complete same-DLL requests average
GPU/CPU 19.96/74.47 ms idle, 64.65/193.20 ms scrolling and 21.11/79.83 ms local
edit; significant tails remain. This compares delivery routes, not live-game FPS.
The final candidate passes exact native UI/camera/config-off ownership and 30
independent frames with no native redraws (21.05 ms request mean). Sampled
contiguous headroom is 1.18 GiB; no player memory setting is required.

Current staged DLL:
`61440a6b13fd8773cd02033868258aa809c38999fe7cd729006154e3f6eb587a`.
Ordinary `INSTALL.bat` uses it; rollback `7d598ac5…` is preserved. Live confirmation
of this repair remains pending. Missing-content work remains milestone 1's next
architectural responsibility; general nonblocking camera publication stays open.
No new hooks, gameplay cadence changes, installation or game launch.

### Worker-side GPU buffer creation checkpoint

**Completed:** the three worker-prepared natural-terrain layers (terrain, decal,
mountain) now have their GPU vertex buffer created on the same worker thread that
packs them, not on the foreground render thread. `TerrainSurfaces` gained an
opaque `std::shared_ptr<void>` buffer handle plus per-layer offsets so the
platform-agnostic compiler header (verified bit-exact against the foreground path
by a standalone Mac test) still never names a D3D11 type; the concrete
`ID3D11Buffer` and its `CreateBuffer`/`Release` calls live only in the renderer
source, behind a small `attach_terrain_vertex_buffer` step run right after
compilation, in both the worker job and the foreground fallback. The device is
created without `D3D11_CREATE_DEVICE_SINGLETHREADED`, so cross-thread
`CreateBuffer` is spec-safe. Index-buffer handling (shared-grid reuse and the
per-tile combined buffer for edge tiles) is unchanged; only vertex bytes for
these three layers move off the foreground path.

**Measured:** wall-clock `geometry_ms`/`request_ms` in the whole-frame benchmark
swung 15–20% between repeated runs of the *same* binary on the Parallels VM, so
that comparison was inconclusive at 64 samples. The deterministic,
run-to-run-identical signal is foreground GPU upload bytes: across 63 dense-
scrolling steps, the control DLL appended 2,428,414 bytes to the foreground
upload accumulator (GPU-resident route) versus 798,450 bytes for the candidate —
a **67% reduction**, reproduced exactly across two independent candidate runs.
The same 67% reduction appears on the CPU-comparison route (4,856,828 → 1,596,900
bytes). `python3 Renderer/renderer.py build`'s embedded tests and
`python3 -m unittest Renderer.native.test_content_preparation` (including the
worker/foreground parity test) both pass; the full native benchmark still passes
every contract (exact pixels/ownership, immutable map, GPU/CPU fallback).

**Next unfinished responsibility:** foreground ground/cliff/city generation and
adoption remain the larger, previously-identified piece of "missing-content"
scrolling cost — this change only removed buffer-creation cost for content
already migrated to the worker. Wall-clock evidence was too noisy on this VM to
confirm the byte-count win's effect on total request time; a lower-variance
timing method (or more samples) would help before deciding whether to chase
further buffer-creation offloads versus migrating ground-layer generation itself,
which is still the bigger remaining foreground cost per the original diagnosis.
