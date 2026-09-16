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

### Bug-repair checkpoint: input, zoom HUD and retained animation

Source fixes unify native mouse picking while keeping clip queries and city-work
input native, and correct the city-HUD attachment point. The user approved the
input and map-UI hooks; all are active. The city HUD now uses its specific
coordinate call, and paired ordinary/army calls place status, selection cursors
and civilization markers. The global city-anchor inlead and all four zoom
bookkeeping fields are removed; `Unit_tick_anim` passes native offsets unchanged.
The GOG injection compile and 65 focused checks pass, including byte/ABI evidence,
actual table wiring and coordinate parity across supported zooms. See the patch
ledger for exact symbols, addresses and signatures.

Retained map samplers now own their copied inputs until composition releases them.
The existing DLL fails the new unpublished-replacement test; candidate
`03e148cf394e1a5b485f3d5609a97da30c97cba00dbc4a28346a94b1b86a0447` passes it,
resource-only idle (Horses and Cattle), native UI/partial transfer/config-off oracles
and 30 independent frames with zero native draw calls. This is a correctness fix,
not a performance improvement claim. Cattle retained composition uses 32.1 MiB;
the final independent request mean is 20.69 ms (desktop completion 29.63 ms).
Local reproducer receipt: `native/build/gpu-composition/298f6135c9894b3da828ba2886d9314a`;
Final connected Cattle result:
`native/build/gpu-composition/42e7de0b371e4d89a755c9552d801d28`.
The exact tested candidate is staged for ordinary `INSTALL.bat`, with the prior
DLL preserved under `native/build/rollback/<sha256>/`. No game was installed or
launched. The pending live checkpoint covers short
left/right click, held drag, repeated Z with city HUD, and idle Cattle without a
working unit. The architectural next responsibility above is unchanged.
