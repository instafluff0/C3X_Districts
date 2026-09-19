# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content; nearby preparation helps
without making every possible destination image a prerequisite for fast movement.
Full detail, authored animation and Civ III's gameplay/state authority remain
unchanged. The zoom-owned map-overlay extension below is a planned rendering
ownership change, not a second visibility, selection, or pathfinding system.

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
valid static color/depth. Feed results into the existing native compositor. Until
the explicit zoom-owned tactical-overlay cutover below, keep native overlays;
preserve unit ordering and terrain occlusion exactly, and keep static UI reusable.

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

## Water effects: roadmap placement

Requested addition (2026-09-17): bring C3X water closer to the Civ VI target —
ocean ripples/swell, directional river flow, shoreline wave/foam activity,
reflections of nearby scene geometry, refraction, sun glint and moon/night
response, coast/lake/ocean color-depth variation, ship-wake VFX, and general
time-of-day response. Most of this is not new scope from zero; placement below
reflects what is already built versus what is genuinely gated on milestones 1–2.

**Already implemented, no milestone dependency:** sun/moon direction and color,
water Fresnel/specular response, day-night glint transition, and coast/lake/
ocean color-depth variation shipped in M6.4/I13A (`environment_lighting_and_
ambient_effects.md`) and are already driving the production water shader.
Refraction and further color/opacity tuning are the same kind of shader-only
work and can be layered in opportunistically whenever the water shader is next
touched, without waiting on any milestone below.

**Gated on milestone 2 (animation as a direct scene operation):** open-water
ripples/swell, directional river flow, and turning on shoreline wave/foam
activity all require continuous per-frame motion that milestone 2 exists to
provide safely. `ocean_wave_findings.md`'s own production recommendation says
so explicitly: doing this today means invalidating/rebuilding all terrain
every frame, which it calls unacceptable. Shoreline waves are further along
than the others — a full candidate already passed 207 integration tests and
is pixel-identical to the current baseline when off (`ocean_wave_findings.md`,
"Quieter shoreline spacing") — so it is not new design work, only enablement.
**Use it as milestone 2's first real validation workload** (selectively
redraw only the animated water band over reusable static color/depth) instead
of a synthetic one; extend the same "animate only what changed" mechanism to
open-water ripples and river flow once it is proven on shoreline waves.

**Gated on both milestone 1 and milestone 2:** reflections of nearby scene
geometry (mountains, buildings) need the whole visible scene's resolved
color+depth as an input (milestone 1's GPU-submitted output) and need
selective redraw when the camera or reflected content changes, not full
static reuse (milestone 2). Sequence this after both land, using the existing
per-city `environment_refresh::Reflection` scratch-target plumbing as the
starting point rather than a new reflection pipeline from scratch.

**Independent of this roadmap's milestones:** ship wake/spray is unit-attached
VFX, the same category as M7.5 attached effects (flames/smoke/steam), not
core water-shader work. It can be scheduled whenever effects work is picked
up, without waiting on milestones 1–4.

## Zoom-owned map overlays: roadmap placement

Requested addition (2026-09-17): because custom rendering owns the three-level
map projection, the renderer must eventually draw the map-plane fog/unseen
territory treatment, selected-unit highlight/cursor, and pathfinding/route
visualization rather than scaling Civ III's versions on top. This is a visual
ownership extension only. Civ III remains authoritative for tile visibility,
viewer changes, selection, hover/targeting context, pathfinding, movement costs,
route/turn semantics, picking, and gameplay; the renderer consumes copied
visibility and tactical-overlay records and never reimplements those rules.

**Milestone 1 — visibility data and static fog pass:** make captured per-tile
visibility/fog/unseen state (already represented by the visible-scene
`visibility_mask` and `fog_status`) an explicit dependency of the final map
output. Add a renderer fog/unseen coverage pass after terrain/objects and before
tactical overlays, with correct map clipping, wrapping, and all 128/160/192 tile
widths. Its standalone/replay tests must cover revealed, fogged, unseen, and
visibility-edge tiles. Do not suppress Civ III's fog yet: live replacement waits
for the atomic publication and invalidation contract in milestone 3.

**Milestone 2 — direct tactical-overlay pass:** add a cheap dynamic pass for
the selected-unit highlight/cursor and the already-computed route visualization.
Capture semantic primitives such as the selected/hovered anchor, route segments,
turn breaks and reachable/blocked indicators from Civ III's authoritative
interaction state or its established draw inputs; do not infer a route from map
data. Keep these pass inputs separate from unit-body animation, preserve depth
and terrain occlusion where the native presentation requires it, and prove that
hover/selection/path changes do not rebuild static terrain or unit content.

**Milestone 3 — coherent live cutover:** publish pixels, camera/zoom transform,
visibility epoch, tactical-overlay revision and overlay inputs as one compatible
view identity. At the map composition boundary, suppress only the corresponding
native fog/unseen, selected-unit, and route draws once the matching renderer
output is ready; never show a new camera with old fog, a stale route, or duplicate
native/renderer marks. Visibility, selection, route cancellation, scroll/wrap,
zoom, device recovery, config-off, and renderer-failure paths all retain their
native fallback. Audit existing draw/capture seams before proposing any new patch
symbol; this roadmap entry authorizes no speculative CSV change.

**Still native unless separately extended:** unit health/activity/status and
stack HUD, civilization markers, city labels, borders, general map text, broader
selection UI, and all non-map screens. Their current transformed-native treatment
remains in place. The overlay cutover needs focused all-zoom visual comparisons
and executable ownership/invalidation tests before it is accepted; it does not
move deferred wonders or Districts forward.

## Current evidence and implementation handoff

**Milestone 1.2 is complete; 1.3 has a working production path.** One scoped
preparation task per missing tile uses the existing `ContentPreparation`
implementation. It has exclusive query/height/river scratch, owns its dependency
records, copies river-node values, and holds cached grid storage. A frame-local
compile lane reuses its bounded two river pages and allocation capacity across
tiles; each tile resets point caches and the dependency recorder. Captured
observations, world/coast topology and decoded assets are read-only until its
enforced join. Every early return, cancellation and exception joins before
captured locals can be destroyed; no ground job or scratch survives the frame.
This settles topology lifetime without duplicating sampling formulas or
broadening dependency footprints.

The production pickup path prepares exact packed meshes and bounds off the render
owner while city/infrastructure, cliffs and natural terrain progress. Adoption
merges private proofs and uses the existing immutable GPU upload owners without
repacking. Ground grid admission preserves its original dependency union. Legacy
analytic shadows preserve their combined terrain/object packing. No quality,
visibility, overlay, capture, cache budget or native ownership changes.
`C3X_RENDERER_GROUND_WORKERS=0` runs the same compiler synchronously for comparison.
The ready queue is limited to one task/result and the existing 16 MiB ceiling.
Raw meshes are released before publication. `ground-preparation` reports compile
wall time (overlapping other work), join time and ready bytes separately.

A pre-existing cliff extraction bug was exposed by the complete workload: hill
height queries used the private river wrapper's empty asset storage. They now
read the loaded immutable `NaturalData`; query and river state remain private.
The performance control includes the identical one-line correction. The original
crashing control is preserved separately.

**Validation and measured effect (2026-09-19):** x86 candidate build passed;
full current-code integration ran 297 tests (295 passed, two skipped) and passed
scrolling, reduced-zoom scrolling, wrapping and authoritative terrain-edit replays.
Executable tests cover the real packed ground compiler, exact dependency proofs,
cache leases, reused scratch, cancellation, exceptions and callback destruction.
No injected source changed; no staging, installation or game launch occurred.

The matched 1120×1192 fixture uses the preserved 100×100 scene, 963 visible tiles,
eight units, native UI and final screen transfer. Each run records 384 complete
requests: 64 per CPU/GPU route and workload. GPU-route request mean / p95 in ms:

| Workload | Repaired control | Candidate |
| --- | ---: | ---: |
| Stationary animation | 11.31 / 22.31 | 10.46 / 19.85 |
| Dense scrolling | 312.33 / 542.83 | 255.63 / 427.05 |
| Local changes | 12.10 / 36.64 | 12.64 / 45.32 |

Dense-scroll mean request cost fell **18.2%**; mean desktop completion fell from
322.34 to 266.45 ms. A separate matched pair with `--profile` repeated the direction:
365.00 → 320.75 ms request (**12.1%**), 373.11 → 330.38 ms desktop completion.
Do not mix instrumentation settings. Small stationary/local-change differences
are inconclusive. Capture is outside timing; desktop completion is not physical
scanout, and neither pair establishes live-game cadence.

In 62 correlated unprofiled GPU scrolling requests with missing content, foreground
ground setup/join/admission fell 228.26 → 155.95 ms. Candidate compile wall time was
174.77 ms and join time 154.70 ms: these overlap other work and must not be added.
The gain includes bounded river-page reuse and moving exact packing onto the worker;
it is not an isolated threading speedup. Remaining joins still dominate.
The profiled candidate retained at least 1,069.6 MiB sampled contiguous free VA
(control 1,178.1 MiB), above the 512 MiB requirement; samples do not establish
transient peak allocation. Ready ground results peaked at 0.326 MiB overall
(0.284 MiB in scrolling).

All four full-frame runs passed ownership/fallback checks and produced identical
control bitmaps. Receipts verified source/DLL inputs throughout; all 3,324 generated
asset files were rehashed unchanged. The local
[comparison receipt](../native/build/ground-checkpoint/comparison.json) preserves
full distributions, phase/memory samples, fixture/build identities and run paths.
**Retain this implementation; milestone-1 performance acceptance remains open.**

**Next unfinished responsibility:** finish milestone **1.3** by reducing the
remaining ground joins. The current tile-scoped overlap is bounded and correct,
but ground still exceeds the independent work available before adoption. Broader
selected-content preparation needs owned per-job inputs and a frame-scoped source
lease; do not merely add threads to closures borrowing tile locals. Milestone
**1.4**, city/infrastructure extraction, follows that responsibility.
GPU adoption beyond ground packing is still 1.5/1.6; whole milestone-1 performance
acceptance remains open. Dynamic units/water and nonblocking camera publication
remain milestones 2 and 3. Wonders/Districts remain deferred.

[Earlier controls, rejected approaches and extraction findings](history/ground_preparation_before_20260919.md)
are preserved as evidence, not additional approval gates or work queues.
