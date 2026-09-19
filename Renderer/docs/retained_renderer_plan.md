# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content; nearby preparation helps
without making every possible destination image a prerequisite for fast movement.
Full detail, authored animation and Civ III's gameplay/state authority remain
unchanged. The zoom-owned map-overlay extension below is a planned rendering
ownership change, not a second visibility, selection, or pathfinding system.

**Active deliverable: milestone 1.** Its replacement static submission and
GPU-ready world preparation path is implemented and validated through 1.6. The
remaining milestone-1 responsibility is the static visibility/fog/unseen pass
defined below; complete its standalone/replay proof before closing the milestone.
Continue that responsibility without another planning/approval gate. Milestones are connected outcomes, not promised short
passes. [Architecture](renderer_architecture.md) owns the design;
[validation](benchmark_workflow.md) owns measurement and acceptance.

## Current implementation and gaps

| Original responsibility | Implemented foundation | Remaining responsibility |
| --- | --- | --- |
| Persistent world/instances | Camera-independent content, shared assets/forest meshes, revisioned units, GPU-ready ground/terrain/objects, immutable mesh ranges/materials, bounded residency | Nonblocking cold/invalidated views; broader sharing where measured |
| Local validity | Captured appearance/dependency revisions, unit revision/despawn proofs | Explicit visibility dependency for the static fog pass; coherent authoritative change publication |
| Spatial selection | World pass index and native/wrapped occurrences feed replacement static submissions directly | Collected dynamic/unit pass inputs |
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

**1.6 static preparation/submission is complete; full milestone 1 still needs the
static fog/unseen pass.** `world_preparation.h` replaces separate ground/object/
terrain scheduling on the ordinary world path with one bounded selected-tile
queue. Existing worker capacity is shared across tiles. Private query scratch
survives requests; each job compiles ground, terrain and objects and creates one
immutable GPU allocation. Adoption retains its exact ranges and dependencies.
Shared grid indices and underlay ranges remain shared. Object packing preserves
source indices instead of expanding triangles and hashing them again.

The frame lease, cancellation joins, resident admission, materials, controls and
per-component oversized/failure recovery remain intact. Ready CPU/GPU/proof storage
uses the existing 64 MiB budget (16 MiB with world preparation disabled); each
combined result is checked against 16 MiB before allocation. Active compilers and
upload transients are additional. The faster path exposed an older pixel producer
that omitted ground/object layers from shared world owners, losing city shadows;
it now copies every world-owned layer. Shadow/region keys also distinguish mesh
ranges within an owner. No new game patch, native ownership change or presenter.

**Validation (2026-09-19):** 310 full-suite tests (308 passed, two skipped), native
scroll/reduced-zoom/wrap and a new city-retained-scroll replay passed. Six paired
images are byte-identical: cities day/night at 128, cities at 64/160/192, and
infrastructure at 160. Ground-serial, object-serial and cached-grid controls pass;
the ground-serial replay exercises 171 prepared blocks and has zero changed pixels
under the existing tolerance, absolute channel error 71 (not byte-identical).
Six city edits each rebuild two tiles/reuse 385 and match fresh output exactly;
the coastline edit rebuilds 127/reuses 260, also exact. All 3,324 asset files are
unchanged. Candidate verification only: no staging, installation, reference
replacement, injected-code change or live-game test.

**Complete workload:** matched 1120×1192 runs include 963 selected tiles, eight
units, native UI and final transfer; each arm has 384 requests across CPU/GPU
stationary, scrolling and local-change routes. Capture is outside timing; desktop
completion is not scanout or live-game cadence. Primary GPU request mean / p95, ms:

| Fixture / workload | Preserved 1.4 control | 1.6 candidate |
| --- | ---: | ---: |
| Mixed terrain + dense objects, stationary | 16.12 / 43.34 | 16.55 / 49.92 |
| Mixed terrain + dense objects, scrolling | 331.60 / 467.54 | 288.61 / 411.06 |
| Mixed terrain + dense objects, local changes | 25.00 / 50.96 | 20.87 / 49.94 |
| Modern cities + infrastructure, stationary | 15.14 / 40.98 | 20.54 / 56.15 |
| Modern cities + infrastructure, scrolling | 209.67 / 255.98 | 148.94 / 169.43 |
| Modern cities + infrastructure, local changes | 22.27 / 51.19 | 25.28 / 65.54 |

Scrolling improves **13.0% mixed / 29.0% modern**; mean desktop completion improves
341.32 → 298.39 / 219.43 → 159.87 ms. Whole-request results, not worker relocation
alone, support the static submission result. Stationary/local tails vary; no
universal frame-time improvement or frame-budget claim is made.

A reverse-order modern-city repeat confirms scrolling at **207.01 → 151.36 ms**
(p95 257.24 → 170.71). Stationary mean/p95 is 16.00/40.27 → 15.92/41.06;
local changes are 20.77/50.96 → 21.30/58.90. Both stationary arms perform zero
world builds and zero geometry work. Retain both pairs: the scrolling benefit is
repeatable, while stationary/local timing does not establish a general speedup.

Across 62 mixed scrolling misses, foreground ground falls 15.18 → 0.40 ms,
features 22.90 → 6.84 ms and upload 35.23 → 17.64 ms. Modern features fall
33.47 → 1.88 ms and upload 47.02 → 8.64 ms. The new world join is outside those
phase timers: 135.12 ms mixed / 16.82 ms modern. Worker component spans overlap;
they are not additive CPU/GPU totals. The remaining mixed cost is principally
terrain generation/join, with cliff/resource adapters retaining their owners.

All final/repeated arms pass, preserve recorded inputs and produce identical
paired output. Candidate sampled contiguous free VA bottoms at **1,094.1 MiB**
mixed and **533.6 MiB** modern across both runs: above the 512 MiB gate, but the
modern margin is only 21.6 MiB. Transient peaks are not established; preserve the
budget gate for future growth. Selected queues peak at 20.5 MiB mixed / 19.2 MiB
modern with four workers and no rejection, eviction or recovery.

[Comparison receipts](../native/build/world-preparation-checkpoint/comparison.json),
[pixel evidence](../native/build/world-preparation-checkpoint/visual-parity.json) and
[validation details](../native/build/world-preparation-checkpoint/validation.json)
retain input/DLL identities, full distributions, controls and failed diagnostic
attempts. Earlier 1.5 evidence remains in Git and the preserved control snapshot.

**Next: finish milestone 1's visibility dependency and static fog/unseen pass.**
Consume captured visibility/fog state, prove revealed/fogged/unseen edges,
clipping and wrapping at 128/160/192, and order coverage after objects but before
tactical overlays. Keep native fog active until milestone 3's coherent publication
and cutover. Then proceed to milestone 2's direct dynamic scene path, using
shoreline waves as its first map-effect workload; general nonblocking camera
publication remains milestone 3. Wonders and Districts remain M9–M11.
