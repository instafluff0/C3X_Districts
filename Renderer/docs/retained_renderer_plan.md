# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content; nearby preparation helps
without making every possible destination image a prerequisite for fast movement.
Full detail, authored animation and Civ III's gameplay/state authority remain
unchanged. The zoom-owned map-overlay extension below is a planned rendering
ownership change, not a second visibility, selection, or pathfinding system.

**Current technical checkpoint: M2.6 scheduling/reuse and M2.7 automated acceptance
are complete in the candidate.** M1 and M2.1–2.5 remain implemented.
Units stay above all map geometry. Native actions, visibility, controls and unit/UI
ordering remain authoritative. Shore waves, water motion and reflections are on
in every normal performance workload. The accepted water DLL remains staged;
these DLL-only optimizations have not been staged, installed or run inside Civ III.
[Architecture](renderer_architecture.md) owns the design;
[validation](benchmark_workflow.md) owns measurement and acceptance.

## M2.6 / M2.7 current handoff

Candidate: `native/build/m26-unit-sharing-final/C3XRenderer.dll`, SHA-256
`b7d2fad4213c1b75e51576a2687521f050784469f5494235953e013d4d319bf9`.
Prior reflection/timer work was committed as `12d73104`; the user handles pushing.

Completed capabilities:

- Independent frames retain material-pass selections and static shadow-caster
  descriptors. Time changes no longer rediscover static submission state.
  Borrowed mesh records retire with their owning scene; scene/view/light changes
  invalidate the retained recipe.
- Direct unit revisions are collected before map execution. Missing current poses
  enter the existing bounded workers ahead of predictions; GPU execution preserves
  native order. Identical unit body contributions share the existing 192 MiB GPU
  content owner. Each occurrence still applies its own shadow, placement and
  underlay blend; hidden/retired selections cannot authorize a cached pose.
- Retained composition reuses owned result allocations and removes a duplicate
  full-map copy. Opaque city bodies remain in static color/depth. Conservative dry
  water rejection is enabled by default; mirror cells without water receivers are
  omitted. User-facing effects and diagnostic controls remain available.
- Resident reflections and shared reflected tree meshes remain enabled. The timer
  rearms after synchronous work, retaining at least 10 ms for the message pump
  after an overrun. Reassess that guard when M3 makes callbacks nonblocking.

Matched complete native workload, 1120×1192 dense coast, eight units, all effects
on, **384 requests per DLL**, 64 per route/workload. Prior `m26-final` versus this
candidate, GPU request mean / p95 (ms):

| Workload | Prior | Candidate |
| --- | --- | --- |
| Stationary native demand | 11.81 / 15.31 | 11.58 / 16.43 |
| Scrolling | 93.62 / 143.27 | 91.98 / 166.19 |
| Local edit | 16.63 / 65.28 | 21.23 / 71.49 |
| Independent 32-unit animation, 120 frames | 48.08 / 68.95 | 40.54 / 62.08 |

The **15.7% lower mean animation request** is the demonstrated gain. Desktop
completion improves 55.12 → 50.57 ms. Scrolling improvement is inconclusive;
local-edit mean and navigation tails regress in this pair. Do not describe this
as a general navigation or frame-budget win. Earlier profiled 65–72 ms visual and
177 ms scrolling numbers included expensive address-space diagnostics; they are
not comparable to these unprofiled production timings.

All 120 candidate animation intervals prove zero static-world builds/uploads,
static-scene draws/readbacks, mirror builds, wave builds/uploads and water uploads;
all reuse material selections without static-caster recollection. Of 3,840 unit
occurrences, 2,445 reuse a body contribution (64% fewer body raster passes).
Native pixel/erase-bounds, 555/565/full-color ownership, CPU fallback, timer,
config-off drain/reenable and tactical cancellation checks pass. The coastal
control image is pixel-exact to the prior DLL. Sampled contiguous address space
stays above **749.6 MiB**; this is sampled headroom, not a transient allocation bound.
Receipts and interval proofs: `native/build/m27-complete-{control,final}/`.

The same all-visible workload measures **31.36 / 37.58 / 40.54 ms mean** at
**1 / 8 / 32 units**, each over 120 frames (one selected idle unit, remaining units
working). P95 is 41.45 / 51.43 / 62.08 ms. The added 31 units cost 9.18 ms in this
repeated-worker fixture; differing unit assets/actions can have different scaling.
All intervals preserve the zero-static-work proof. Additional receipts:
`native/build/m27-scaling1/` and `native/build/m27-scaling8/`.

The separate 120-frame eight-unit mixed/visibility run passes: one selected idle,
three work loops, two frozen idle and two native-directed actions; 33.41 ms mean,
46.77 ms p95. Fog freezes map animation, hidden units emit no pixels, reveal resumes
motion, and route/selection/grid changes preserve static content.
Evidence: `native/build/m27-mixed8/`; this different workload is not a scaling
comparison against the 32-worker case.

Eight new destinations still cost **0.38–1.74 s**, with no fallback. Initial scene
preparation costs 4.61 s. Four existing world workers prepare current content;
new terrain and foreground joins remain material costs. A separate serialized
probe places 143–433 ms at even a one-pixel CPU readback after sub-millisecond
scene/finish completion queries. On this VM those queries do not establish true
physical GPU completion; the observed delay is at the readback synchronization
boundary, not proof of cheap shader execution. The earlier 4.30 s wait did not
recur, but is not proved eliminated. Evidence: `native/build/m27-distant-final/`
and `native/build/m27-distant-attribution/`. These completed-render/capture
measurements include CPU delivery, not native presentation or physical scanout.

**Next unfinished architectural responsibility: M3 coherent state publication and
nonblocking camera/content updates**, including cold, evicted and invalidated
views. Keep the current scene valid while useful work completes; never present
mismatched camera/visibility/picking. M4 owns a demonstrated cadence budget.
Fast cold jumps, sustained 30/60 FPS and strategic live-game acceptance are not
claimed by M2's independent-animation proof. No new native hook, ABI or patch-table
entry is required.

Verification: the full dependency-selected suite passes **313 tests / two skips**.
The existing native composition, lifecycle and configuration controls pass in all
five completed 120-frame receipts (480 current-candidate frames plus 120 historical
control frames), including frozen/native action cases. Another 32 focused tests
pass, including 126 exact native retained-composition oracles and 120 clock frames. Separate native water witnesses cover beach and rocky coasts,
daylight/moonlight, 48-frame playback, fog/reveal, reset, reflection toggles,
zoom return, wrapped reduced-zoom replay and authoritative local edits. Repeat,
reset, reflection toggle, replay and edit comparisons are exact; the independent
still-water control differs at ten pixels by one channel level, within its existing
rounding contract. Disabling waves leaves water motion active; disabling both
proves a still image. Evidence: `lab/out/integration/m27-{waves-final,rocky-final,
river-lifecycle,wrapped-scroll,local-edits}/`. No renderer appearance, action/visibility
contract, injected source or executable address changed. Strategic live-game
acceptance and staging of this optimization candidate remain pending.

## Current implementation and gaps

| Original responsibility | Implemented foundation | Remaining responsibility |
| --- | --- | --- |
| Persistent world/instances | Camera-independent content, shared assets/forest meshes, revisioned units, GPU-ready ground/terrain/objects, immutable mesh ranges/materials, bounded residency | Nonblocking cold/invalidated views; broader sharing where measured |
| Local validity | Captured appearance/visibility dependencies, immutable map inputs, unit revision/despawn/hidden proofs | Coherent general authoritative change publication |
| Spatial selection | World pass index and native/wrapped occurrences feed replacement static submissions directly | Broader spatial sharing where measured |
| Compatible passes | Compatible static layers, shared material bindings, batched occurrence parameters/uploads, forest instancing, collected poses/shared body contributions, exact native composition | Additional compatible sharing guided by measured cost |
| GPU reuse/output | Resident map color/depth, direct eligible unit geometry, incremental finishing and native composition | Reduce dynamic conversion, replay and full-map work |
| Async integration | Bounded GPU-ready ground/terrain/object preparation with shared worker capacity, selected-work urgency and independent visual timer | Coherent general nonblocking camera/content publication |

Independent frames use the existing HWND presenter without native redraw requests.
Civ III's original gameplay timer is unchanged. Map animation still enters the
existing render orchestration. Eligible units use collected direct scene execution with shared body contributions;
oversized native canvases retain bounded GPU pose compatibility. Further GPU work,
finishing and publication costs remain for the explicit M3/M4 budgets.

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
overlays**; **2.6 scheduling/reuse**; **2.7 acceptance**. The shoreline portion of
2.4 now has resident-scene lifecycle and connected native-frame coverage.

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

**Milestone 2.4 map effects:** shoreline ribbons now execute in the shared dynamic
pass over reusable static color/depth. Time-only updates build/upload no terrain
or ribbon geometry and submit no static scene draws. The existing material,
spacing, 15 Hz sampling, visibility and optional asset controls are preserved.
The earlier full-terrain-rebuild concern in `ocean_wave_findings.md` describes the
pre-retained architecture, not this path. Reflections-enabled rendering keeps its
existing compatibility path; resident reflection work remains below.

**Open water and rivers implemented:** the existing normal textures now provide
overlapping ripples and animated highlights without mesh displacement
or new texture assets. River normals follow renderer-owned immutable tangents,
derived from native connectivity toward water outlets; closed components use a
stable visual sink. Wrapped flow and remote-outlet changes participate in river
page validity. No native patch, gameplay calculation or extra input capture is
needed. Fog freezes the material sample. Time-only frames reuse geometry and
static color/depth; affected translucent forward layers retain their original
order. Existing reflection compatibility remains supported.

**Current follow-up:** see the M2.6/M2.7 handoff above for implementation and
measured limits. Water visuals are accepted and staged; the strategic live-game
check remains distinct from executable verification.

**Current visual refinement:** smaller overlapping ocean ripples use three bounded
CPU phases, with no dominant translating sheet. Sun/moon highlights form a
concentrated broken path using a material-only finite-eye approximation derived
from the authoritative anchors. Shared light direction/intensity remains the
source of that path. The map projection is unchanged, and this adds no texture,
render target, mesh or pass. The ocean optical approximation is not underwater
refraction or a claim about recovered Civ VI shader code. Rivers retain their
connected downstream motion. Motion-off camera changes expire only finished
rasters, keeping the optical path out of stale world-image caches.

Current candidate SHA-256:
`3976b380f5ce0240b16c069e7f00118c8cc6643ffe3b40c22f7a8999f1782655`.
The focused suite passes 125 tests / one skip. Integration covers 302 tests
(300 pass / two skips after updating two extracted-source fixtures for the water
fields; the repaired 26-test subset passes). Windows HDR, visibility and native
CPU handoff checks pass. Day/night, fog/reveal, river
playback, still-scroll/cold and wrapped-camera/cold witnesses pass. Repeat/fog and
camera comparisons are exact; the independent split/static finish differs by
one channel level at one pixel, within the existing rounding budget. Current
previews live under `lab/out/water-refinement/final/`. The user accepted these
visuals on September 19. The exact DLL is staged in `bin/C3XRenderer.dll`, with
matching SHA-256 and a rollback copy recorded in
`lab/out/integration/water-accepted/staging.json`. No install or game launch ran. The current full-workload receipt is
`native/build/water-refinement-workload-final/receipt.json` (pass, inputs unchanged,
no dropped trace lines). The same 1120×1192 dense coast, eight-unit native workload
has 384 requests across CPU/GPU routes, with waves on and reflections off.
This is historical evidence, not the new all-effects performance baseline. GPU request mean / p95: stationary
12.51 / 35.76 ms, scrolling 152.65 / 217.41 ms, local edits 17.73 / 44.86 ms.
The separate one-selected-unit independent workload has 30 frames: 66.76 / 79.62
ms request mean / p95, 76.16 ms desktop mean, zero static submissions and zero
content uploads in every interval. Sampled contiguous VA stays above 948.9 MiB.
The preceding run averaged 66.99 ms independent and 152.12 ms scrolling; this
similar result is not an isolated shader ablation or a frame-budget acceptance.
Capture/setup remain outside timing and desktop completion is not scanout.
`summary.json` alongside the receipt preserves distributions and interval proof.

The first attempt was stopped after its test loop starved its own deadline check
by draining continuously due timers. The witness now checks progress after each
callback, preserving the three-frame/three-second requirement. M2.6 must also
address runtime callback fairness when a visual frame exceeds the 33 ms timer
interval; fixing this witness does not improve production cadence.

**Initial water workload checkpoint (2026-09-19, before the light-path refinement):** candidate SHA-256
`f0917aef1be98b9b0c8c027b25624f1434c849e49e439a5b3e69bb6a6df5befa`.
Same DLL/assets, serial motion-off/on runs, established 100×100 coast at
1120×1192, dense cities/infrastructure, eight units, native UI and shoreline
waves enabled. Each arm includes 384 timed native requests (64 per route and
workload) plus 30 independent visual frames. GPU request mean / p95, ms:

| Workload | Water motion off | Water motion on |
| --- | ---: | ---: |
| Stationary native requests | 13.41 / 35.08 | 11.83 / 35.85 |
| Dense scrolling | 141.36 / 192.03 | 152.12 / 224.08 |
| Local changes | 17.15 / 40.97 | 15.66 / 48.71 |
| Independent visual frames | 57.20 / 64.19 | 66.99 / 73.35 |

Independent frames use one selected unit and retained UI; eight units apply to
the native-request workloads. Independent desktop-completion means are
68.77 / 76.67 ms. All 30 motion-on
intervals contain water samples, zero static scene submissions and zero content
uploads. Pixel/ownership, native fallback, independent timer and bounded unit
composition checks pass. Sampled minimum contiguous VA is 1036.3 / 1025.6 MiB,
above the 512 MiB floor; sampling does not bound transient peaks. Input identities
remain unchanged and buffered traces have no dropped lines.

This pair measures added effect cost, not an equivalent-image speedup. Small
native-request differences are inconclusive; the extra ~9.8 ms independent-frame
mean and ~10.8 ms scrolling mean are disclosed costs. Capture/setup are outside
timing, desktop completion is not scanout, and current cadence remains below
the eventual frame-budget objective. The slower optional reflection compatibility
route remains supported, not promoted as the performance path.

Receipts/counters: `native/build/water-motion-workload/comparison.json` and its
`coast-off` / `coast-on` receipts. Water Lab witnesses cover daylight, moonlight,
48-frame playback, cold/repeat/time return, fog/reveal, the independent still
control and reflection compatibility. Wrapped scrolling and authoritative local
edits match independent cold redraws exactly. Category tests: 135 passed / one skipped;
three focused water-coverage tests pass. Earlier failed diagnostic runs remain
preserved: duplicate compatibility water was fixed; the small 32×32 Lab fixture
hit an outside-map alpha oracle mismatch at the large benchmark viewport, so the
established 100×100 benchmark was used without weakening its native pixel oracle.
The water visuals are now accepted/staged; live-game evidence remains pending.

The 2.6 candidate now retains the existing planar reflections of nearby geometry
using the original `environment_refresh::Reflection` scratch and the current
scene/dependency owners. Broader reflection algorithms are not prerequisites for
optimizing this enabled production workload.

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

**Milestone 2.5 — direct tactical-overlay pass:** implemented in the candidate for
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
patch dependency. The audited GOG grid hook and two route hooks are now recorded
in the [patch ledger](civ3_patch_dependency_ledger.md); the
[tactical contract](tactical_overlay_contract.md) records copied inputs and lifecycle.

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

**2.3 committed and pushed:** `9835c9a8` was committed on Mac and pushed through
Windows. Units remain above all map geometry, retaining pose-local shadows,
self-depth, authored actions, fog eligibility, native composition and controls.
Unused map-depth capture/provenance and the map-region cache are removed. Its
measurements remain in Git and `unit-composition-checkpoint/workload-comparison.json`.

**2.4 shoreline capability implemented:** retained wave cells feed the existing
shadow/foam/resource dynamic pass and damage union. The renderer restores static
samples, draws the affected ribbons and finishes that band. Waves no longer
exclude automatic shared-scene admission. No new surface owner, scheduler, API,
native hook or patch-table entry is introduced; `required_user_action: []`.
Reflections retain their configuration and existing compatibility renderer;
resident scene measurements below use reflections off. Open-water motion and
river flow are unfinished. No binaries were staged/installed, game launched or
reference images replaced.

**2.4 evidence:** the [shoreline findings](ocean_wave_findings.md#milestone-24-resident-scene-checkpoint)
preserve its 295-test/nine-replay verification, complete-request distributions,
independent-frame measurements and unchanged-asset identities.

**2.5 tactical capability implemented:** the existing composition history now owns
copied selection rings, native route segments/turn strings and gray grid edges.
The white broken ellipse has rotating inward markers; analytic antialiasing and
restrained shadows sharpen the route and label. Native selection, pathfinding,
turn arithmetic and `MapGrid_Flag` remain authoritative; no Ctrl+G listener or
second route model exists. Native erase/copy operations retire the same immutable
history. The DLL does the rendering; injected code supplies the audited seams.
Three GOG additions are recorded in the [patch ledger](civ3_patch_dependency_ledger.md#milestone-25--tactical-draws-gog)
under the user's patch-table authorization; other-build addresses remain zero.
Before composition admission/with an older DLL, native cursor/route drawing
continues. Config-off calls the original functions.

**2.5 verification:** 276 tests (**274 passed / two skipped**), approved injected
compile/injection smoke test and four production replays pass. Additional connected
native fixtures pass at 64/128/160/192 and with fog. Route cancellation/grid-off
restore the prior image exactly. Twelve independent selection samples execute
without native draw events, static terrain submissions or new unit-content builds;
the existing 128 MiB retained-history limit includes copied primitive capacity.
The preview is a synthetic scene through the real JGL/DLL path, not live gameplay.
The connected category uses the established 100×100 camera fixture: a 32×32
preview world also fails its wider prepared-camera assertion with overlays off.
No assertion was relaxed.

**Complete-workload effect:** same candidate, 1120×1192 dense modern-city coast,
reflections/waves off, 8/32 unit bodies, native UI and final transfer. Each request
contains one selected marker, one two-segment route/turn label and the viewport
grid when enabled. Six serial runs contain 2,304 timed CPU/GPU requests; the
32-unit pair was repeated in reverse order after a variable local-change tail.
GPU mean / p95 milliseconds (64 samples per 8-unit cell; 128 per 32-unit cell):

| Units / workload | Overlays off | Overlays on |
| --- | ---: | ---: |
| 8 / stationary requests | 15.01 / 39.98 | 14.02 / 41.71 |
| 8 / dense scrolling | 141.87 / 215.36 | 139.54 / 190.74 |
| 8 / local changes | 18.08 / 41.22 | 17.21 / 37.09 |
| 32 / stationary requests | 19.96 / 51.28 | 21.17 / 49.95 |
| 32 / dense scrolling | 144.77 / 214.94 | 146.55 / 205.32 |
| 32 / local changes | 23.88 / 60.79 | 27.74 / 74.01 |

This adds a measured feature cost, not a speedup: stationary/local medians rise
about 1.8–2.8 ms; the lower 8-unit means reflect tail variation. The 32-unit
local-change mean difference was 6.12 ms initially and 1.59 ms in the repeat
(3.86 ms combined). Full distributions remain in the receipts. Native route
calculation and capture setup are outside this semantic replay harness; packet
preparation, drawing and final transfer are inside. CPU/GPU images differ when
tactical visuals are enabled, so their ratio is not an equivalent-image speedup.
The 12-frame independent correctness excerpt averages 24.64 ms including its
112.89 ms cold first replay (remaining 11: 16.62 ms); it is not an FPS/cadence pass.

All six runs have complete traces and unchanged inputs; all **16,102** asset/input
files remain unchanged. Minimum sampled contiguous VA is **648.57 MiB**, above
the 512 MiB floor; sampling does not bound transient peaks. The
[combined receipts](../native/build/tactical-checkpoint/workload-comparison.json)
retain identities, original/repeated distributions and scope limits. Candidate
SHA-256: `6c3b4072c00ab51621b6e423575d2e4aa1f94b4eadf04cd83a21dedcd199fc16`.
[Context preview](../lab/out/tactical-overlays/connected-route.png),
[selection motion excerpt](../lab/out/tactical-overlays/connected-motion.mp4) and
[grid preview](../lab/out/tactical-overlays/connected-grid.png) await visual acceptance.

**Current follow-up:** the M2.6/M2.7 handoff above supersedes this historical
2.5 checkpoint. M3 owns general nonblocking publication; wonders/Districts remain
M9–M11. Tactical visual acceptance and the strategic live-game checkpoint remain
separate from automated lifecycle and performance evidence.
