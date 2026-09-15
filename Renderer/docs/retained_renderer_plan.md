# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content; nearby preparation helps
without making every possible destination image a prerequisite for fast movement.
Full detail, authored animation and native ownership remain unchanged.

**Active next deliverable: milestone 1.** The user released the documentation
pause. On implementation continuation, proceed through the production path below;
choose routine implementation details without another planning/approval gate.
The first attribution check belongs inside that work, not a new open-ended
research/tooling phase. Milestones are connected outcomes, not four promised short
passes. [Architecture](renderer_architecture.md) owns the design;
[validation](benchmark_workflow.md) owns measurement and acceptance.

## Current implementation and gaps

| Original responsibility | Implemented foundation | Remaining responsibility |
| --- | --- | --- |
| Persistent world/instances | Camera-independent tile content, shared assets, revisioned unit instances, bounded residency | Broader compact shared-mesh instances and reusable GPU submission descriptions |
| Local validity | Captured appearance/dependency revisions, unit revision/despawn proofs | Preserve complete validity through migrated representations; refine authoritative change publication where useful |
| Spatial selection | World pass membership, native occurrences/anchors, clipping/wrapping | Selected inputs that directly feed compatible submissions across migrated content |
| Compatible passes | Terrain/object grouping, forest instancing, GPU unit shadows, exact native composition | Broader instancing/state grouping and collected dynamic/unit submission |
| GPU reuse/output | Resident admitted map/poses, static color/depth, incremental finishing and native composition | Direct dynamic scene execution; reduce per-pose and full-map work where measured |
| Async integration | Bounded content/pose/view preparation and independent visual timer | Coherent general nonblocking camera/content publication; UI-thread stalls remain |

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

Code checkpoint: `c1360ea9`. Tested DLL staged for ordinary `INSTALL.bat`:
`3a024a15dd79843a8da7cccc8657474674a31ac26a9fa403165e5c67ae88bf8a`.
The user installs/launches; no environment setup is required. Existing staging
permission persists, but this documentation task does not stage or launch anything.

Automated independent frames and native UI/ownership tests pass; live validation
of decoupling remains pending. Latest whole-request GPU means: 9.52 ms idle,
51.19 ms scrolling, 11.45 ms local change versus control 8.95 / 47.58 / 10.73 ms.
Map work accounts for 44.1 of 51.2 ms scrolling, without resolving CPU versus GPU
attribution. Independent frames average 17.60 ms request / 29.04 ms desktop
completion. Decoupling is a capability gain, not a demonstrated foreground speedup.

[Checkpoint evidence](history/retained_renderer_checkpoints_20260915.md) preserves
sample counts, tails, tests, binary/receipt identities and expensive findings.
First implementation deliverable: make selected static map pass inputs feed reusable
GPU rendering descriptions and compatible submissions, extending existing world
validity and worker ownership. Use one dense scrolling fixture to distinguish
construction, selection, submission and downstream rendering cost, then choose
the representative shared-instance conversion within that path. Compare the full
replacement on idle animation, scrolling and a local edit against the preserved
control. This defines the deliverable; exact batch boundaries, worker count and
GPU recording strategy remain measured implementation choices. Complete this
connected deliverable, adapt mechanisms when evidence warrants it, and update this
handoff with capabilities, measured effects and the next unfinished responsibility.
No additional milestone or output-helper detour is implied.
