# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content; nearby preparation helps
without making every possible destination image a prerequisite for fast movement.
Full detail, authored animation and Civ III's gameplay/state authority remain
unchanged. The zoom-owned map-overlay extension below is a planned rendering
ownership change, not a second visibility, selection, or pathfinding system.

**Current technical checkpoint: 1.6, 2.1 and 2.2 implemented; the 2.2 regression
has been substantially reduced by bounded GPU-input and allocation reuse.** The
complete workload still costs more than the pre-2.2 pose-cache control; Milestone 2
performance acceptance is not claimed. Visual acceptance and the strategic
live-game check remain pending. **Next: 2.3, shadows, occlusion and composition.**
[Architecture](renderer_architecture.md) owns the design;
[validation](benchmark_workflow.md) owns measurement and acceptance.

## Current implementation and gaps

| Original responsibility | Implemented foundation | Remaining responsibility |
| --- | --- | --- |
| Persistent world/instances | Camera-independent content, shared assets/forest meshes, revisioned units, GPU-ready ground/terrain/objects, immutable mesh ranges/materials, bounded residency | Nonblocking cold/invalidated views; broader sharing where measured |
| Local validity | Captured appearance/visibility dependencies, immutable map inputs, unit revision/despawn/hidden proofs | Coherent general authoritative change publication |
| Spatial selection | World pass index and native/wrapped occurrences feed replacement static submissions directly | Direct dynamic/unit execution inputs |
| Compatible passes | Compatible static layers, shared material bindings, batched occurrence parameters/uploads, forest instancing, exact native composition | Collected dynamic/unit execution; additional sharing guided by cost |
| GPU reuse/output | Resident map color/depth, direct eligible unit geometry, incremental finishing and native composition | Shared dynamic occlusion/composition; reduce conversion, replay and full-map work |
| Async integration | Bounded GPU-ready ground/terrain/object preparation with shared worker capacity, selected-work urgency and independent visual timer | Coherent general nonblocking camera/content publication |

Independent frames use the existing HWND presenter without native redraw requests.
Civ III's original gameplay timer is unchanged. Map animation still enters the
existing render orchestration. Eligible units use direct scene execution; native
overlap or unavailable scene inputs retain GPU pose compatibility. The current
conversion and per-unit execution costs are not the cheap scene-frame endpoint.

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
replace the temporary unit-over-map ordering with shared terrain occlusion in
2.3, preserve native UI ordering, and keep static UI reusable.

Shared pose buffers, GPU deformation/skinning and grouped targets are candidates
where they remove measured work. Do not force all units into one surface or alter
source playback just to simplify batching.

**Done:** idle animation does not rebuild world content or rediscover static
submission state; animated-unit scaling avoids repeated independent setup/finish
work where compatible. Verify frozen units, authored work loops and native actions.

The requested substeps are: **2.1 immutable dynamic inputs** (implemented; see
[contract](dynamic_scene_input_contract.md)); **2.2 direct unit poses** (implemented; see
[contract](direct_unit_scene_contract.md)); **2.3 shadows,
occlusion and composition**; **2.4 map effects**, shoreline waves first; **2.5 tactical
overlays**; **2.6 scheduling/reuse**; **2.7 acceptance**. Existing optional wave tests
are compatibility checks, not early enablement of 2.4.

**2.3 required occlusion fix (user screenshot, 2026-09-19):** a unit on the tile
behind a mountain or other tall foreground object currently draws over it. Correct
unit-body occlusion using compatible world/scene depth and the actual foreground
geometry, replacing 2.2's separate near/far depth bands. Tile order alone is
insufficient for partial overlap. Verify units both behind and in front of relief,
vegetation and other existing tall map objects, including movement across the
occlusion boundary, animation, zoom, scrolling/wrap and fog. Keep unit HUD/cursor
ownership separate. This is a 2.3 acceptance requirement, not deferred polish.

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
**Use it as milestone 2.4's first real map-effect validation workload** (selectively
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

**Milestone 1 — visibility data and static fog pass:** API 18 captures normalized
visibility/fog/unseen state from the native viewer rules as an explicit dependency
of final output. The older `visibility_mask` is a traversal mask, and `fog_status`
alone omits other native visibility fields. Renderer coverage follows terrain/objects and precedes
tactical overlays, with correct map clipping, wrapping, and all 128/160/192 tile
widths (and reduced 64). Its standalone/replay tests cover revealed, fogged,
unseen and visibility-edge tiles. The user's explicit direct-hook authorization
advances fog suppression into this checkpoint: API 18 uses the existing coherent
synchronous map publication and exclusive custom-map failure contract. The wrapper
returns in custom mode and calls native fog unchanged when off. This does not
claim milestone 3's general nonblocking publication. Units under fog are omitted;
explored resource/effect motion uses stable still samples.

**Milestone 2.5 — direct tactical-overlay pass:** add a cheap dynamic pass for
the selected-unit highlight/cursor, the already-computed route visualization, and
the optional thin gray tile grid.
Capture semantic primitives such as the selected/hovered anchor, route segments,
turn breaks and reachable/blocked indicators from Civ III's authoritative
interaction state or its established draw inputs; do not infer a route from map
data. Keep these pass inputs separate from unit-body animation, preserve depth
and terrain occlusion where the native presentation requires it, and prove that
hover/selection/path changes do not rebuild static terrain or unit content.

**2.5 gridline support (requested 2026-09-19):** Civ III remains responsible for
its global grid setting and Ctrl+G handling. Do not add a keyboard listener,
shortcut interception, or a separate toggle state. Identify the native grid draw
function and use its authoritative enabled state or invocation within the current
map draw, whichever its call contract supports. Keep the hook minimal: in custom
mode pass the required draw/state inputs to the renderer DLL and return without
native drawing; otherwise call the original function unchanged. The DLL draws
the tile boundaries using native anchors and the current projection. Switching
on/off must add/remove only the grid, without stale lines, double drawing or
terrain rebuilds. Verify clipping, fog/unseen coverage, zoom and wrapping. This
cutover belongs to 2.5 using existing coherent publication; milestone 3 extends
it to nonblocking views. Audit the actual function before recording any concrete
patch dependency; no grid hook or CSV edit is needed for this roadmap update.

**Milestone 3 — coherent live cutover:** publish pixels, camera/zoom transform,
visibility epoch, tactical-overlay revision and overlay inputs as one compatible
view identity. Extend the existing fog publication contract to nonblocking views
and suppress corresponding selected-unit and route draws only once matching renderer output
is ready; never show a new camera with old fog, a stale route, or duplicate
native/renderer marks. Visibility, selection, route cancellation, scroll/wrap,
zoom and device recovery retain coherent ownership. Config-off retains native
rendering; custom-on map-plane failures preserve exclusive custom-map handling.
Audit existing draw/capture seams before proposing any new patch
symbol; this roadmap entry authorizes no speculative CSV change.

**Still native unless separately extended:** unit health/activity/status and
stack HUD, civilization markers, city labels, borders, general map text, broader
selection UI, and all non-map screens. Their current transformed-native treatment
remains in place. The overlay cutover needs focused all-zoom visual comparisons
and executable ownership/invalidation tests before it is accepted; it does not
move deferred wonders or Districts forward.

## Current evidence and implementation handoff

**2.2 optimization pass implemented; performance acceptance remains open.**
The pre-optimization checkpoint is `5c48ea46` (committed on Mac, pushed from
Windows). Current improvements are in the working tree. Nothing was staged or
installed; Civ III was not launched and reference images were not replaced.

1.6's static submissions/fog and 2.1's immutable inputs remain intact: hidden units
are suppressed, explored resources/effects freeze, and Civ III owns actions,
visibility and anchors. The [dynamic input contract](dynamic_scene_input_contract.md)
and [patch ledger](civ3_patch_dependency_ledger.md) retain those contracts.
No injected/CSV changes; `required_user_action: []`.

**Work eliminated:** eligible unit geometry still executes over resident map
color/depth, without finished CPU/GPU pose caching or terrain resubmission. Its
existing pose owner now reuses exact GPU vertices and shadow inputs under a
**192 MiB** combined input allowance. Matching retired allocations are recycled.
Raw map captures retain reusable idle allocations within their existing **64 MiB**
budget; active leases cannot be overwritten. Restoration, extraction, hardware
resolve and native composition follow the body/shadow footprint. Scene and
compatibility conversion scratch are separate, eliminating route-switch resize
churn. The native raster viewport and erase bounds stay unchanged; the shared
working attachment remains **75 MiB** in this witness (96 MiB admission cap).
A cropped raster viewport was rejected after the new oracle found a one-channel
rounding difference. See the [direct scene contract](direct_unit_scene_contract.md).

**Verification:** **292 tests: 290 passed / two skipped**, plus all six production
replays (scroll/reduced/wrap, resources, day/night units). Each unit replay covers
288 body cases and 582 action/held-endpoint checks with unchanged terrain. Final
visible/fogged independent frames and timer transport pass. The strengthened
real-map oracle admits **all 128 cases directly** at 750/1000 projection, with exact
555/565 and full-color pixels through clipping, placement, light/zoom changes,
**56 GPU-input builds, 72 hits and 36 allocation reuses after eviction**. This
post-measurement test strengthening uses the **same measured DLL**. Existing
sample-scale, native ownership, config-off and CPU-barrier checks pass.

**Complete workload:** serial 1120×1192 dense-modern-city comparisons include map,
8/32 units, native UI and final transfer. Each run has **384 timed CPU/GPU requests**,
64 samples per route/workload. All source/DLL identities stayed fixed per run and
**16,102 asset/input files** were unchanged across the final comparison. GPU whole
request mean / p95, milliseconds (the regression column preserves the earlier
checkpoint measurements; the other two columns are a fresh paired comparison):

| Units / workload | Fresh pre-2.2 control | Recorded 2.2 regression | Optimized |
| --- | ---: | ---: | ---: |
| 8 / stationary animation | 13.92 / 37.29 | 26.27 / 78.13 | 18.32 / 44.04 |
| 8 / dense scrolling | 145.26 / 170.49 | 164.68 / 197.71 | 153.20 / 182.11 |
| 8 / local changes | 23.06 / 78.79 | 39.33 / 103.11 | 24.86 / 78.04 |
| 32 / stationary animation | 19.30 / 52.77 | 67.50 / 158.21 | 23.89 / 55.14 |
| 32 / dense scrolling | 150.70 / 180.57 | 222.36 / 326.02 | 160.36 / 188.52 |
| 32 / local changes | 21.71 / 63.16 | 89.46 / 206.90 | 29.56 / 80.00 |

At 32 units this removes **65% / 28% / 67%** of recorded request time, but still
adds **4.6 / 9.7 / 7.8 ms** against the fresh control. Optimized mean desktop
completion is **26.50 / 162.94 / 34.02 ms** at eight units and
**32.74 / 171.02 / 37.95 ms** at 32. This is not scanout or live-game FPS.
An earlier repetition of the same optimized DLL averaged **27.59 / 172.01 / 27.28 ms**;
four consecutive slow map-preparation frames raised its scroll p95 to **344.66 ms**.
Keep that variability visible; the final comparison does not establish a tail budget.

All optimized traces are complete, including every timed GPU interval. Final
32-unit intervals contain **3,360 direct draws**, **432 GPU-input builds / 2,928
hits**, **1,406 idle-region reuses**, and 336 compatibility builds / 2,448 hits;
no unit-body readbacks or composition uploads. Admission mix varies with raw
source-generation availability; the earlier repetition had 4,928 direct draws.
Peak charged GPU inputs are **186.1 MiB**, raw regions **64.0 MiB**. Sampled minimum
contiguous VA is **624.7 MiB** in the final optimized 32-unit run, **592.4 MiB** in
its repetition, and **595.9 MiB** across final controls, above the 512 MiB floor.
Transient peaks between samples remain unmeasured. The old control's 8 MiB trace
still truncates detailed 32-unit diagnostics; its 384 frame timings are complete.

[Combined receipts](../native/build/unit-scene-checkpoint/optimization/workload-comparison.json)
preserve full distributions, run/binary identities, counters, rejected experiments,
memory and trace limits. `lab/out/integration/units.json` holds full verification.

**Next: 2.3 shadows, occlusion and composition.** Establish shared receiver/depth
membership and compatible unit submissions, explicitly fixing units appearing
over foreground mountains/tall objects. Replace the separate depth bands and
reduce the remaining per-unit restore/draw/finish/composition work; retain the
complete-workload control comparison and 512 MiB floor. Preserve native UI order,
visibility and action authority. General scheduling/reuse remains 2.6, acceptance
2.7; shoreline waves start 2.4, tactical overlays including native-setting-driven
gridlines 2.5, nonblocking publication milestone 3, wonders/Districts M9–M11.
