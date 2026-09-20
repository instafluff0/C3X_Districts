# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content; nearby preparation helps
without making every possible destination image a prerequisite for fast movement.
Full detail, authored animation and Civ III's gameplay/state authority remain
unchanged. The zoom-owned map-overlay extension below is a planned rendering
ownership change, not a second visibility, selection, or pathfinding system.

**Current technical checkpoint: 1.6, 2.1 and 2.2 implemented; 2.3 implemented and verified.**
The user's final policy keeps units above all map geometry. The direct path now
removes map-depth capture/provenance instead of implementing terrain occlusion.
Milestone 2 performance acceptance and the strategic live-game check remain open.
**Next: 2.4, map effects beginning with shoreline waves.**
[Architecture](renderer_architecture.md) owns the design;
[validation](benchmark_workflow.md) owns measurement and acceptance.

## Current implementation and gaps

| Original responsibility | Implemented foundation | Remaining responsibility |
| --- | --- | --- |
| Persistent world/instances | Camera-independent content, shared assets/forest meshes, revisioned units, GPU-ready ground/terrain/objects, immutable mesh ranges/materials, bounded residency | Nonblocking cold/invalidated views; broader sharing where measured |
| Local validity | Captured appearance/visibility dependencies, immutable map inputs, unit revision/despawn/hidden proofs | Coherent general authoritative change publication |
| Spatial selection | World pass index and native/wrapped occurrences feed replacement static submissions directly | Direct dynamic/unit execution inputs |
| Compatible passes | Compatible static layers, shared material bindings, batched occurrence parameters/uploads, forest instancing, exact native composition | Collected dynamic/unit execution; additional sharing guided by cost |
| GPU reuse/output | Resident map color/depth, direct eligible unit geometry, incremental finishing and native composition | Reduce dynamic conversion, replay and full-map work |
| Async integration | Bounded GPU-ready ground/terrain/object preparation with shared worker capacity, selected-work urgency and independent visual timer | Coherent general nonblocking camera/content publication |

Independent frames use the existing HWND presenter without native redraw requests.
Civ III's original gameplay timer is unchanged. Map animation still enters the
existing render orchestration. Eligible units use direct scene execution; oversized
native canvases retain bounded GPU pose compatibility. The current
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
keep units above all map geometry per the final user decision in 2.3, preserve
native unit/UI ordering, and keep static UI reusable.

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

**2.3 final scope (user reversal, 2026-09-19):** units always draw above map
geometry, whether on, behind or in front of a mountain, forest or building. The
previous terrain-occlusion requirement is withdrawn, including the intermediate
same-tile exception. Keep existing pose-local shadows, native unit/UI order,
actions, visibility and controls. Remove the unused map-depth coupling and verify
exact body/native composition across overlap, movement, animation, zoom and fog.
This does not add surrounding-world shadow receivers or arbitrary-geometry unit
shadow casting. No deferred terrain-occlusion task is implied by this policy.

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

**2.3 complete under the user's final always-on-top policy.** Units draw above
all map geometry, with native unit/UI ordering, pose-local shadows, full self-depth,
authored actions, fog eligibility and quality controls intact. The earlier terrain
occlusion implementation was withdrawn. Surrounding-world shadow reception and
unit casting onto arbitrary geometry are not claimed.

The optimized 2.2 checkpoint `cc550142` was committed on Mac and pushed through
Windows before this work. 2.3 is a DLL-only working-tree change: no API, injected
source or patch-table edits; `required_user_action: []`. Candidate verification
did not stage/install binaries, launch Civ III or replace references.

**Work eliminated:** no raw map color/depth capture, source-generation lease,
provenance transport through native copies, or separate map/unit depth bands.
The direct pass clears only its footprint, draws cached GPU pose inputs into a
transparent attachment and uses existing exact native composition. The 64 MiB
map-region cache and its retired tests are removed. GPU pose inputs remain capped
at 192 MiB; the work attachment uses 75 MiB here (96 MiB cap). Oversized zoom
canvases retain bounded GPU compatibility. Native raster coordinates and hardware
MSAA resolve remain exact; no terrain resubmission or CPU unit roundtrip is added.
See the [direct scene contract](direct_unit_scene_contract.md).

**Verification:** 291 tests, **289 passed / two skipped**, plus all six production
replays. Day/night each covers 288 body cases and 582 action/held-endpoint checks.
There are 98,304 exact resolve comparisons, 126 retained-composition oracles and
120 independent frames. Connected native-screen fixtures pass at 64/128/160/192
and with fog, including nine mountain/forest/building placements, 128 direct
555/565/full-color comparisons, native UI/partial transfer, timer transport,
visibility freeze/reveal, ownership and CPU barriers. The
[placement contact sheet](../native/build/unit-composition-checkpoint/placements.png)
was inspected; its CPU-oracle images match the actual native output exactly.

**Complete workload:** serial matched optimized-2.2 control and 2.3 candidate,
1120×1192 dense modern cities, 8/32 units, native UI and final transfer. Each run
contains 384 timed CPU/GPU requests, 64 per route/workload. All identities stayed
fixed per run; all 16,102 asset/input files match the optimization baseline and
remain unchanged. GPU whole-request mean / p95 in milliseconds:

| Units / workload | Optimized 2.2 control | 2.3 |
| --- | ---: | ---: |
| 8 / stationary animation | 14.94 / 26.36 | 13.97 / 37.95 |
| 8 / dense scrolling | 157.68 / 206.31 | 156.05 / 194.69 |
| 8 / local changes | 29.34 / 80.87 | 17.35 / 41.04 |
| 32 / stationary animation | 27.69 / 72.87 | 19.50 / 45.26 |
| 32 / dense scrolling | 166.90 / 202.72 | 165.88 / 278.13 |
| 32 / local changes | 29.79 / 81.69 | 30.89 / 61.43 |

The reverse-order 32-unit repeat measured control **27.91 / 174.02 / 38.42 ms**, candidate **22.12 / 164.73 / 25.63 ms**
(stationary / scroll / local-change means). Its scroll p95 was 264.77 / 189.32 ms (control / candidate).
Dense scrolling remains map-dominated. In the first pair, both binaries have
slow map preparations at the same three scroll steps; an additional candidate
map spike moves its p95 upward. Unit submission mean falls from 8.34/6.43/7.22
to 3.81/4.33/4.16 ms, but local-change total mean rises slightly and desktop
completion can vary. Phase intervals include submission/waits, not isolated GPU
time. These results do not establish a tail budget, physical scanout or live FPS.

The first 32-unit candidate has **6,144 direct timed draws, 768 GPU-input builds /
5,376 hits**, no compatibility draws, no map captures, no body readbacks and no
composition uploads. All 192 timed GPU intervals have complete counter coverage.
Peak input charge is 186.1 MiB and map-region charge is zero. Sampled minimum
contiguous VA is **845.2 MiB at eight units / 654.1 MiB at 32**, above the 512 MiB
floor. The expanded 32-unit control reaches **444.9 MiB**, below that floor;
its successful pixel/timing run is not a memory acceptance pass. The reverse-order
repeat samples 1001.9 MiB candidate / 632.8 MiB control; address-space layout and
fragmentation vary between processes. Samples do not
bound transient peaks. Enlarged visual fixtures still exercise GPU compatibility.

[Combined receipts](../native/build/unit-composition-checkpoint/workload-comparison.json)
preserve both pairs/repeats, identities, full distributions, counters and limits.
`unit-composition-checkpoint/integration.json` preserves full verification;
candidate SHA-256 is
`2432744745feecefc7054da2ea782c1c4afd3d5fa0ecfd9636ca29d6f4485548`.

**Next: 2.4 map effects, shoreline waves first.** Extend the renderer-owned dynamic
input/pass lifecycle while preserving authoritative visibility, frozen explored
content and complete-workload measurement. Tactical overlays and native-setting-
driven gridlines remain 2.5 (no Ctrl+G listener), scheduling/reuse 2.6, milestone
acceptance 2.7, nonblocking publication M3 and wonders/Districts M9–M11. Strategic
live-game acceptance remains pending; 2.3 does not certify Milestone 2 performance.
