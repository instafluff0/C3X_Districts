# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content; nearby preparation helps
without making every possible destination image a prerequisite for fast movement.
Full detail, authored animation and Civ III's gameplay/state authority remain
unchanged. The zoom-owned map-overlay extension below is a planned rendering
ownership change, not a second visibility, selection, or pathfinding system.

**Current technical checkpoint: 1.6, 2.1 and 2.2 implemented.** Eligible unit
geometry now draws over valid resident map color/depth. **The complete workload
regresses against the preserved pose-cache control; this is architectural delivery,
not Milestone 2 performance acceptance.** Visual acceptance and the strategic
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

**2.2 is technically implemented; its measured performance regression remains
unresolved. Milestone 2 acceptance is not claimed.** Visual acceptance and the
strategic live-game checkpoint remain pending. Nothing was staged or installed;
Civ III was not launched and reference images were not replaced.

1.6's static submissions/fog and 2.1's immutable inputs remain intact: hidden units
are suppressed, explored resources/effects freeze, and Civ III owns actions,
visibility and anchors. The [dynamic input contract](dynamic_scene_input_contract.md)
and [patch ledger](civ3_patch_dependency_ledger.md) preserve those rules. Earlier
measurements remain in Git and `native/build/visibility-checkpoint/`.

**2.2:** map-source/coordinate provenance travels through native composition.
Eligible prepared unit geometry draws over resident map color/depth without
terrain submission or a finished CPU/GPU pose cache. Conservative body/shadow
coverage preserves native erase bounds. Unchanged retained dependencies reuse
completed composition; native overlap, stale samples, eviction or failed admission
use the existing GPU pose cache. Raw regions are capped at **64 MiB**; the shared
scene target at **96 MiB** accommodates authored 4× sampling (**75 MiB** in the
witness). Hardware MSAA resolve and conversion scratch preserve exact native
pixels. Shared world-depth occlusion and shadow receivers remain 2.3. See the
[direct scene contract](direct_unit_scene_contract.md). No injected/CSV edits;
`required_user_action: []`.

**Verification:** **292 tests: 290 passed / two skipped**, plus scroll/reduced/wrap,
resource playback and day/night unit replays. Each unit replay covers 288 body
cases and 582 action/held-endpoint checks with unchanged terrain. GPU checks cover
98,304 exact native/full-color pixels, all sample scales, clipped regions and
circular raw color/depth transport. Final visible/fogged independent frames,
timer transport, 555/565/UI ordering, config-off and CPU barriers pass.
`lab/out/integration/units.json` and the combined receipts below retain details.

**Complete workload:** serial preserved-control/candidate comparisons at
1120×1192 include dense modern cities, native UI and final transfer. Each variant
has **384 timed CPU/GPU requests**, 64 samples per route/workload. Source/DLL
identities stayed fixed per run; **16,102 asset/input files** were unchanged across
the final comparison. GPU whole-request mean / p95, milliseconds:

| Units / workload | Preserved 2.1 control | 2.2 candidate |
| --- | ---: | ---: |
| 8 / stationary animation | 14.32 / 37.16 | 26.27 / 78.13 |
| 8 / dense scrolling | 149.85 / 188.30 | 164.68 / 197.71 |
| 8 / local changes | 18.90 / 63.03 | 39.33 / 103.11 |
| 32 / stationary animation | 16.41 / 37.49 | 67.50 / 158.21 |
| 32 / dense scrolling | 156.31 / 215.82 | 222.36 / 326.02 |
| 32 / local changes | 27.00 / 65.66 | 89.46 / 206.90 |

This is a regression, especially at 32 units. Candidate mean desktop completion
is **34.26 / 175.45 / 47.32 ms** at eight units and **75.40 / 231.46 / 98.33 ms**
at 32. Capture is outside timing; desktop completion is not scanout or live-game
FPS. Repeated supersampled scene execution/conversion is more expensive here than
cached poses. Shared composition and dependency reuse must address it before
Milestone 2 acceptance; do not promote this as a performance improvement.

Eight-unit timed GPU intervals contain **1,048 direct draws**, 144 compatibility
builds and 344 hits, with **zero region evictions**, body readbacks or composition
uploads; raw regions peak at **21.0 MiB**. At 32 units, regions peak at **64.0 MiB**
and at least 2,784 direct draws / 1,347 evictions were observed. Its detailed trace
hit the existing 8 MiB limit: operation counts are lower bounds, while all 384
frame timings are complete. Sampled contiguous VA stays above **837.9 MiB for the
candidate / 654.5 MiB across the controls**, exceeding the 512 MiB floor;
continuous transient peaks remain unmeasured.

[Combined receipts](../native/build/unit-scene-checkpoint/workload-comparison.json)
retain run identities, distributions, memory and trace limits. Intermediate
failures remain preserved: shader-only resolve lost exact pixels; whole-canvas
proofs rejected valid draws; oversized captures churned; a 64 MiB work cap rejected
authored sampling. The final path uses hardware resolve, tight regions and the
measured 96 MiB work cap.

**Next: 2.3 shadows, occlusion and composition.** Establish shared receiver/depth
membership and compatible unit composition, explicitly fixing units appearing
over foreground mountains/tall objects. Preserve native UI order, visibility and
action authority. Address the measured conversion/replay/region costs without
hiding them behind isolated timings. Scheduling/reuse remains 2.6, acceptance 2.7;
shoreline waves start 2.4, tactical overlays including native-setting-driven
gridlines 2.5, nonblocking publication milestone
3, and wonders/Districts M9–M11.
