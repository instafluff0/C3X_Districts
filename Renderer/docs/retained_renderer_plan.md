# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content; nearby preparation helps
without making every possible destination image a prerequisite for fast movement.
Full detail, authored animation and native ownership remain unchanged.

**Active deliverable: milestone 1.** Its replacement static submission path is
implemented and validated; its whole-request performance criterion remains open.
Continue the missing-content responsibility identified below without another
planning/approval gate. Milestones are connected outcomes, not promised short
passes. [Architecture](renderer_architecture.md) owns the design;
[validation](benchmark_workflow.md) owns measurement and acceptance.

## Current implementation and gaps

| Original responsibility | Implemented foundation | Remaining responsibility |
| --- | --- | --- |
| Persistent world/instances | Camera-independent content, shared assets/forest meshes, revisioned units, immutable mesh ranges and material bindings, bounded residency | Remove remaining foreground construction of missing content; broaden mesh sharing where useful |
| Local validity | Captured appearance/dependency revisions, unit revision/despawn proofs | Preserve complete validity through migrated representations; refine authoritative change publication where useful |
| Spatial selection | World pass index and native/wrapped occurrences feed replacement static submissions directly | Collected dynamic/unit pass inputs |
| Compatible passes | Compatible static layers, shared material bindings, batched occurrence parameters/uploads, forest instancing, exact native composition | Collected dynamic/unit execution; additional sharing guided by cost |
| GPU reuse/output | Resident admitted map/poses, static color/depth, incremental finishing and native composition | Direct dynamic scene execution; reduce per-pose and full-map work where measured |
| Async integration | Bounded content/pose/view preparation, selected-work urgency and independent visual timer | Complete GPU-ready content preparation; coherent general nonblocking camera/content publication |

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

**Completed:** selected static inputs now use compiled material bundles, bounded
parameter streams and compatible layer submissions. Changed mesh ranges share
immutable allocations within their existing residency owners. The old per-layer
allocation and unconditional layer-flush paths are replaced. Current worker demand
is explicitly prioritized. Full detail, dependency validity, native ownership and
config-off behavior are preserved; no new hooks or visual concessions were added.

**Measured:** the full-detail 1120×1192 connected fixture includes map, eight units,
native UI and final transfer (capture remains outside timing). Latest original DLL
versus candidate GPU request means / p95, 64 samples per workload:

| Workload | Original (ms) | Candidate (ms) |
| --- | --- | --- |
| Stationary animation | 9.68 / 15.89 | 9.08 / 15.03 |
| Dense scrolling | 56.06 / 99.90 | 55.88 / 104.41 |
| Local change | 11.62 / 33.86 | 11.40 / 37.89 |

Scrolling mean is effectively unchanged; its median improved but tails did not.
This is **not a demonstrated whole-request speedup**. Submission/allocation work
fell, while content preparation still averages 31.2 ms of the 55.9 ms request.
Outer-halo CPU speculation regressed scrolling and was removed. Exact control
pixels, native UI/ownership and independent-frame tests pass; sampled contiguous
32-bit headroom is 1.47 GiB. These fixtures do not establish live-game acceptance.

**Next unfinished responsibility:** complete preparation of missing world content
into GPU-ready geometry, bounds, dependency proofs and upload ranges, leaving
bounded adoption on the GPU owner. Existing terrain workers still return raw mesh
vectors, and object assembly/packing still runs in the foreground. Extend the
existing compilers and validity owners; do not add another speculative coverage
queue or output-helper detour. Keep milestone 1 performance acceptance open until
the complete workload improves. Direct dynamic/unit scene execution remains
milestone 2; general coherent nonblocking native camera publication remains 3.

Candidate DLL `9b7a72e935d6b39100767f3f544287c7352e94431d63cc5fda641e25a7e8e722`
is staged for ordinary `INSTALL.bat`; the user installs/launches. No installation
or game launch was performed. Original rollback DLL and all intermediate results
remain preserved. [Submission checkpoint](history/selected_submissions_20260915.md)
records exact controls, receipts, tests and rejected mechanisms;
[earlier evidence](history/retained_renderer_checkpoints_20260915.md) remains valid.
