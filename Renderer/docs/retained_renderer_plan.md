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

**Milestones 1.2–1.5 are complete; milestone-1 performance acceptance remains open.**
`object_preparation.h` now prepares city bodies/ground/paving, bridges, routes,
mines, farms, huts/camps, fallback buildings and walls on a bounded worker.
It selects, constructs, packs and creates immutable vertex/index buffers;
foreground adoption retains their ranges directly. Legacy/non-world profiles keep
their existing path. Resources, forests and native overlays retain their owners.
The representation remains exact retained meshes, not new source-mesh instancing.

Inputs own capture scalars and borrow immutable assets/observations under a frame
lease. Query scratch is private; world/coast/river and absent route-neighbor proofs
enter the existing cache. Lighting, materials, forest exclusions and fallback are
preserved. Cancellation/error exits join all readers. Ready CPU/GPU/proof storage
is capped at 16 MiB. Expanded-geometry preflight permits 32 MiB of estimated
payload (container/packing overhead is additional); private river scratch permits
two pages. One object lane shares the terrain allowance with ground; completion
notifications return lanes without waiting for foreground progress. Callbacks are
unregistered/joined before local owners disappear. `C3X_RENDERER_OBJECT_WORKERS=0`
uses the same compiler synchronously; oversized/failed work uses that recovery path.

**Validation (2026-09-19):** current x86 candidate/full integration passed 309 tests
(307 passed, two skipped), plus native scroll/reduced-zoom/wrap replays.
Infrastructure checks passed 129 tests (128 passed, one skipped). Five paired
images are byte-identical: city day/night at 128, city gameplay at 192, city at 64,
and infrastructure at 160. Six city appearance edits each rebuild two tiles/reuse
385 and exactly match fresh output, including neighboring forest exclusions.
The established coastline edit with objects rebuilds 127/reuses 260 with zero
pixel difference. All 3,324 asset files remain unchanged. No staging, installation,
reference replacement, injected-code/patch change or live ownership expansion.

**Complete workload:** matched 1120×1192 runs include 963 visible tiles, eight
units, native UI and final transfer; each arm has 384 requests, with identical
profiling/budgets. Capture is outside timing; desktop completion is not scanout
or live-game cadence. GPU request mean / p95, milliseconds:

| Fixture / workload | 1.4 control | 1.5 candidate |
| --- | ---: | ---: |
| Mixed terrain + dense objects, stationary | 16.83 / 56.61 | 12.84 / 36.39 |
| Mixed terrain + dense objects, scrolling | 293.91 / 410.18 | 333.49 / 464.41 |
| Mixed terrain + dense objects, local changes | 17.73 / 46.91 | 21.02 / 65.97 |
| Modern cities + infrastructure, stationary | 14.58 / 35.12 | 16.38 / 38.12 |
| Modern cities + infrastructure, scrolling | 213.12 / 279.08 | 223.82 / 274.23 |
| Modern cities + infrastructure, local changes | 24.35 / 72.44 | 22.57 / 47.78 |

**No overall speedup claim:** scrolling is 13.5% slower on mixed terrain and 5.0%
slower with modern cities. Across 62 mixed scrolling misses, foreground feature
work falls 20.64 → 11.33 ms and upload 29.67 → 27.60 ms, but terrain preparation/wait
rises 101.79 → 139.05 ms. The modern fixture uses `--dense-city-case 0,3,1,1` and
averages 25.16 rigid plus 7.74 ground/paving city parts per miss. Its upload falls
47.82 → 10.07 ms and city assembly 3.73 → 0.19 ms; however, 89.56 ms of the 91.15 ms
feature span waits for the producer, which supplies about 8.76 MB of GPU data.
Mean desktop completion is 222.57 → 232.99 ms. Overlapping spans are not additive.

Evidence changed scheduling: returning lanes only at foreground tile boundaries
gave 357.82 ms mixed scrolling (serial prepared-object control: 325.61 ms).
Producer-completion notifications reduced the final candidate to 333.49 ms without
extra worker allowance. Repeated 1.4 controls were stable at 293.54 and 293.91 ms.

Final candidate sampled contiguous free VA stayed above **1,015.6 MiB** on mixed
terrain and **735.5 MiB** with modern cities, exceeding the 512 MiB gate; transient
peaks are not established. Correlated scrolling had no object/ground rejection,
eviction or recovery. Successful arms preserved recorded inputs and exact paired
output. One modern-city attempt failed the native timer precheck before samples;
its unchanged retry passed. The failed receipt is retained and excluded from timing.

[Comparison receipts](../native/build/object-preparation-checkpoint/comparison.json),
[pixel evidence](../native/build/object-preparation-checkpoint/visual-parity.json) and
[validation details](../native/build/object-preparation-checkpoint/validation.json)
retain identities, distributions, workload classifications, memory and failed/repeated
attempts. The prior handoff and 1.4 DLL are preserved in the local control directory.

**Next: 1.6.** Resolve the whole-request regressions before milestone-1 acceptance.
Profile the critical path across object production/join, terrain preparation,
ground/terrain index uploads and remaining adapters. Use measured need to choose
shared preparation capacity or upload batching while preserving bounded memory,
source leases, resident ownership and exact output. Re-run both complete workloads
and their stationary/local-change cases. Dynamic units/water remain milestone 2,
coherent nonblocking camera publication milestone 3, and wonders/Districts M9–M11.
