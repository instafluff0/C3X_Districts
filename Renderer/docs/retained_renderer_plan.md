# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content; nearby preparation helps
without making every possible destination image a prerequisite for fast movement.
Full detail, authored animation and Civ III's gameplay/state authority remain
unchanged. The zoom-owned map-overlay extension below is a planned rendering
ownership change, not a second visibility, selection, or pathfinding system.

**Current technical checkpoint: 1.6 and 2.1 implemented.** The replacement
static submission/preparation path now includes final fog/unseen coverage, and
independent visual frames consume renderer-owned immutable dynamic inputs.
Visual acceptance and the strategic live-game check remain pending; automated
verification is not visual promotion. **Next architectural responsibility: 2.2,
direct unit-pose rendering over resident map color/depth.** Milestones are connected
outcomes, not promised short passes. [Architecture](renderer_architecture.md) owns
the design; [validation](benchmark_workflow.md) owns measurement and acceptance.

## Current implementation and gaps

| Original responsibility | Implemented foundation | Remaining responsibility |
| --- | --- | --- |
| Persistent world/instances | Camera-independent content, shared assets/forest meshes, revisioned units, GPU-ready ground/terrain/objects, immutable mesh ranges/materials, bounded residency | Nonblocking cold/invalidated views; broader sharing where measured |
| Local validity | Captured appearance/visibility dependencies, immutable map inputs, unit revision/despawn/hidden proofs | Coherent general authoritative change publication |
| Spatial selection | World pass index and native/wrapped occurrences feed replacement static submissions directly | Direct dynamic/unit execution inputs |
| Compatible passes | Compatible static layers, shared material bindings, batched occurrence parameters/uploads, forest instancing, exact native composition | Collected dynamic/unit execution; additional sharing guided by cost |
| GPU reuse/output | Resident admitted map/poses, static color/depth, incremental finishing and native composition | Direct dynamic scene execution; reduce per-pose and full-map work where measured |
| Async integration | Bounded GPU-ready ground/terrain/object preparation with shared worker capacity, selected-work urgency and independent visual timer | Coherent general nonblocking camera/content publication |

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

The requested substeps are: **2.1 immutable dynamic inputs** (implemented; see
[contract](dynamic_scene_input_contract.md)); **2.2 direct unit poses**; **2.3 shadows,
occlusion and composition**; **2.4 map effects**, shoreline waves first; **2.5 tactical
overlays**; **2.6 scheduling/reuse**; **2.7 acceptance**. Existing optional wave tests
are compatibility checks, not early enablement of 2.4.

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

**1.6 and 2.1 are implemented and technically verified. Visual acceptance and the
strategic live-game checkpoint remain pending.** Nothing was staged or installed,
Civ III was not launched, and fixed reference images were not replaced.

The static world path uses one bounded selected-tile preparation queue and shared
worker capacity for ground, terrain and objects. Jobs create immutable GPU mesh
ranges; adoption retains those ranges and dependencies. Source indices replace
triangle expansion and rehashing. Shared grid/underlay data, cancellation joins,
local invalidation, controls and per-component failure recovery remain intact.
Ready CPU/GPU/proof storage retains its 64 MiB budget (16 MiB when disabled), with
16 MiB per combined result; active compilers and upload transients are additional.

API 18 adds normalized native visibility to output/view eligibility without making
visibility part of geometry identity. Final map coverage blends soft black unseen
edges and a half-opacity gray explored overlay after objects, before native units
and tactical/UI draws. Raw retained color/depth remains reusable. GPU delivery
uses the completed map texture without another readback. The authorized GOG fog
inlead, `Map_Renderer_draw_fog` at **0x4C4EF0**, returns in custom mode and forwards
unchanged when off. See [exact ABI/build support](civ3_patch_dependency_ledger.md).
The existing `Unit_tick_anim` suppresses hidden bodies, markers, cursors and status;
it does not advance/cancel actions. Hidden DLL selections are retired. Explored
resources and optional waves use stable still samples and request no animation;
visible instances resume authored motion. Gameplay state remains authoritative.

**2.1 replaces mutable visual-map `ProspectiveView` copies with const owned map
records.** Resources, native anchors, visibility, topology, environment, clocks and
publication identity travel together. Retained fronts own their inputs; reset and
configuration invalidate sampling without prematurely releasing budget charges.
Map admission is capped at **16 MiB**. Existing revisioned `UnitInstances` remains
the unit/action owner; no parallel gameplay or animation-state owner was added.
[Dynamic input contract](dynamic_scene_input_contract.md) records validity,
clock, budget, failure and lifecycle rules.

**Verification:** 316 tests, **314 passed / two skipped**; approved injected compile;
scrolling, reduced zoom, world wrap and retained-city replay all passed. Independent
CPU/GPU fog equations agree within one channel level over **3,295,332 pixels** at
64/128/160/192, including clipping, wrap and reset. All four city zoom replays keep
fogged motion exact over three advancing samples and a cold reset. Reveal changes
pixels with **zero geometry builds/uploads**; the GPU replay and optional wave
fixture also prove resumed motion. Native GPU/CPU unit oracles prove hidden bodies
leave pixels unchanged and return empty bounds. Actual native composition tests
preserve UI ordering, ownership, config-off and explicit CPU barriers with zero
execution readbacks on admitted GPU chains. Independent resource/unit/timer tests
pass; immutable-input peak there is **1,599,024 bytes**, with zero rejected records.

The full category receipt is `lab/out/integration/cities.json`. Fog evidence is in
`native/build/visibility-checkpoint/` (`final-z*`, `waves-final`, byte audit and
workload comparison); independent GPU proof is run `34e02af0027f42c6ade55945ee538cbb`.
Failed diagnostic attempts are preserved: an off-screen resource fixture, stale
upload revision, wave-control configuration mismatch, and extracted-test mocks.
Their corrected proofs pass; they are not counted as successful initial runs.

**Prior 1.6 complete comparison:** matched 1120×1192 workloads include 963 selected
tiles, eight units, native UI and final transfer; 384 requests per arm. Versus the
preserved 1.4 control, scrolling improved **13.0% mixed terrain / 29.0% modern cities**.
A reverse-order modern repeat confirms **207.01 → 151.36 ms** request mean,
**257.24 → 170.71 ms** p95. Stationary/local timing does not establish a general
speedup. Sampled contiguous VA remained at least **1,094.1 MiB mixed / 533.6 MiB
modern**; the narrow modern margin and unmeasured transient peaks remain limits.
All 3,324 asset files were unchanged. Detailed comparisons and pixel/control proofs
remain in `native/build/world-preparation-checkpoint/` and Git history.

**Final API 18 + 2.1 workload:** the same 1120×1192 dense-modern fixture, eight
units, native UI and final transfer; 384 timed CPU/GPU requests per variant,
64 samples per route/workload. Both runs pass with unchanged source/DLL/assets
identities. GPU whole-request mean / p95, milliseconds:

| Workload | Fully visible | Visible + explored + unseen |
| --- | ---: | ---: |
| Stationary animation | 13.61 / 34.90 | 13.38 / 20.55 |
| Dense scrolling | 156.69 / 231.30 | 134.59 / 157.95 |
| Local changes | 22.09 / 54.70 | 20.17 / 50.72 |

Mean desktop completion is 21.77 / 166.95 / 31.72 ms fully visible and
20.52 / 144.73 / 28.58 ms with fog, in the same workload order. Visibility changes
the eligible motion workload; these variants do not isolate fog cost or establish
an overall speedup. The visible scrolling tail varies versus the earlier 1.6
repeat; no cadence/frame-budget acceptance is claimed. Capture remains outside
timed requests; desktop completion is not scanout or live-game FPS.

Immutable inputs peak at **2,791,024 bytes (2.66 MiB)**, with zero rejected records.
The largest fog-record upload is **7,184 bytes**. Sampled minimum contiguous VA is
**838.6 MiB fully visible / 994.9 MiB fogged**, above the 512 MiB floor; transient
peaks remain unmeasured. The fully visible output skips the coverage draw/upload.
Receipts: `4e85447d66ae4abd9e0dc0d5ea1868c1` (visible) and
`c51c3da633c942ebbdf7dde4487a885c` (fogged), under `native/build/gpu-composition/`.
[Combined measurements](../native/build/visibility-checkpoint/workload-comparison.json)
retain both CPU/GPU distributions. The initial unconfirmed VM dispatch is preserved;
Windows confirmed no running benchmark before the successful serial retry.

**Next: 2.2 direct unit-pose rendering over resident map color/depth.** Preserve
visibility eligibility, native action cursors, frozen behavior, painter order,
terrain occlusion and existing composition controls while replacing per-pose
setup/finishing where compatible. Follow with 2.3 composition, then shoreline waves
as 2.4's first map-effect workload. General coherent nonblocking publication remains
milestone 3; wonders and Districts remain M9–M11.
