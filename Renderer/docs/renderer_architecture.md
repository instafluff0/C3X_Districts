# Renderer architecture

## Destination

Civ III publishes authoritative scene changes; the renderer retains GPU-ready
world content and owns visual frames between those changes. Moving the camera
should mostly select visible occurrences, update projection and submit compatible
passes. It should not normally reconstruct geometry or require a pre-rendered
image of the destination. Unknown, evicted or newly changed content still has a
real preparation cost, which must be bounded and reported.

The [roadmap](retained_renderer_plan.md) is the sole current status and milestone
sequence. [Validation](benchmark_workflow.md) defines evidence, not another queue.
These documents describe responsibilities; they do not mandate a new engine,
ECS, class hierarchy, API migration or permanent residency of every world mesh.

## World readiness and arbitrary navigation

An unchanged distant destination must be drawable from persistent world content;
its first camera visit must not be the event that discovers and constructs all
of its appearance. Extend `ScenePublication`, `CapturedScene`, `ResidentContent`
and their existing compilers with explicit coverage/revision/readiness. Full-map
topology is not a substitute for authoritative remote city, resource, improvement
and visibility-dependent appearance. Copy that input through the native caller;
background workers never retain a `Map*`, `Tile*` or other game-owned pointer.
Offscreen mutation coverage must be proved before remote records are called fresh.

Use compact shared meshes/materials and rigid instances alongside compiled
regional/deformed geometry. Native projection remains authoritative per visible
occurrence. Preparation follows initialization and actual dependency changes;
residency is a bounded representation of that prepared world. GPU eviction need
not imply reconstructing topology, procedural layout or city composition, but
retaining CPU/streamable backing has its own measured memory/storage cost. Do not
assume the current expanded geometry for every tile/zoom fits in a 32-bit process.
A residency claim must include all consumers: visible color, caster/receiver and
reflection context, water, units, materials and native composition resources.

Track separately: unknown/stale authority, known but uncompiled content,
compiled but nonresident content, and current resident content. A fast resident
subset must not obscure readiness holes across the normal supported world.
Readiness never grants visibility: unseen tiles stay black, fog hides units and
explored animation remains frozen. No lower-detail proxy or stale-camera frame
is an authorized shortcut.

The roadmap's M3.8 contract owns whole-world readiness and coherent native camera
cutover for every trigger: scrolling, minimap, zoom, selected-unit centering,
action following and other native programmatic moves. Automatic centering has the
same latency objective as manual navigation while preserving native destination
and gameplay ordering. M4 closes remaining measured frame costs. Completion must reach a safe
caller-thread adoption boundary promptly without depending solely on a 66 ms
Animator opportunity. Preserve native action progression and centering semantics;
do not solve visual latency by running extra gameplay updates. The target is
<33 ms p95 from input or native camera decision to coherent display for each
trigger class on nominal 100 × 100 Standard maps (5,000 actual tiles), with
viewport, density and hardware declared. Huge maps (12,800 actual tiles) remain
a capacity target with separate latency results. Disclose cold/evicted/edit costs
and initialization time separately.

This is C3X's intended architecture, not a verified description of Civ VI's
runtime. Installed ArtDef/package evidence establishes asset data, not Civ VI's
world-residency policy or camera scheduling. D3D11 already supports shared
per-vertex data plus per-instance inputs through
[DrawIndexedInstanced](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-drawindexedinstanced);
an API migration is not a prerequisite for that representation.

The native presenter uses a [composition swap chain](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgifactory2-createswapchainforcomposition)
attached to the existing Civ III HWND, replacing the former blt-model presenter.
It introduces no new window or competing presenter. Ordinary HWND Present can
[wait on the message-pump thread](https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-idxgiswapchain-present);
a composition swap chain has no HWND and supports independent ambient delivery.
The visual is [below native child windows](https://learn.microsoft.com/en-us/windows/win32/api/dcomp/nf-dcomp-idcompositiondevice-createtargetforhwnd),
and is detached before native GDI resumes. This is not a flip-model HWND chain,
which would disable ordinary GDI even after release. Native ownership, exact
composition and config-off recovery remain required tests. Windows 8+ is
required for this presentation path; the production target is Windows 11.

## Ownership and data flow

```text
Civ III: authoritative changes, camera, visibility, actions and native UI
    -> copied scene changes and view observations
    -> persistent world + shared assets + local dependency revisions
    -> selected visible/wrapped occurrences and pass contributors
    -> compact instance inputs + reusable rendering descriptions
    -> compatible pass submissions on the GPU owner
    -> resident color/depth + native composition -> existing HWND presenter
                 ^ renderer visual clock and bounded preparation workers
```

An asset owns reusable geometry/material/clip data. A world instance owns stable
identity, placement, state, dependencies and bounds. A visible occurrence adds
native projection, wrapping and current eligibility. GPU allocations and finished
images are replaceable representations of that content, not its identity.

| Responsibility | Extend these existing owners | Required behavior |
| --- | --- | --- |
| Scene bridge | `injected_code.c`, versioned renderer API | Capture owned values; publish lifecycle/content changes and exact view/visibility identity. Never pass game pointers to workers. |
| Render world and validity | `render_core/captured_scene.h`, `world_topology.h`, resident content and `unit_instances.h` | Retain identity across views/eviction; invalidate actual dependencies; prevent removal/reuse from reviving stale handles. |
| Assets and compilation | Pack loaders, fidelity adapters, `ResidentContent`, shared natural/resource meshes | Compile changed content once; retain source meshes plus compact instance state where compatible; retain unique/deformed geometry where needed. |
| View and pass selection | `geometry_draws.h`, `world_pass_index.h`, contributor indices | Select native-authorized wrapped occurrences, with separate caster/receiver/reflection bounds; deduplicate without granting visibility. |
| GPU execution | Renderer worker, `draw_cached_geometry`, instance streams, scene color/depth and unit passes | Consume explicit pass inputs, group compatible submissions, batch parameter uploads and avoid redundant state discovery/binding. |
| Composition and presentation | `gpu_composition_session.h`, `retained_composition.h`, native presenter | Preserve exact native operation order, partial transfers and UI ownership; present complete compatible results through the existing window. |

Paths in the table are relative to `Renderer/native/` unless stated otherwise.
The table identifies owners to extend; it does not claim every responsibility
already has complete implementation. See the roadmap's current gaps.

## Representation and submission

Repeated rigid objects should reference shared meshes/materials and carry small
placement/variant records. Terrain, transitions and connectivity can retain
regional geometry; genuinely terrain-conforming objects can retain deformation.
Cities retain component choices and transforms. Source-specific import stays
offline; runtime consumes generic C3X packs. Camera motion must not reseed layout.

Compile stable pipeline/material/mesh descriptions when their dependencies change.
Per-frame selection supplies occurrences, parameters and pass membership. Matching
materials alone do not authorize sorting: preserve cutout, transparent, decal,
water, depth, painter-order and native overlay contracts. Group only compatible
submissions. Keep full detail and authored animation speed; projected-size removal,
LOD or other new appearance changes require separate visual consideration.

The static production path borrows compiled material bundles and mesh ranges from
their existing asset/content owners. Changed meshes share an immutable allocation
within each residency owner; camera-specific and shared world content never share
an allocation across independent eviction lifetimes. Selected occurrences retain
their native projection. Adjacent pass layers share a submission while their
receiver-shadow pages fit; draw parameters upload in bounded batches. Their single
64 KiB stream appends into untouched ranges when the driver supports
[dynamic constant-buffer NO_OVERWRITE](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_map),
discarding only on wrap. Unsupported drivers keep per-batch DISCARD. Offsets and
sizes retain 256-byte alignment, and queued ranges are never overwritten. Color,
reflection and shadow consumers use the same mesh ranges and established order.
Unique terrain/city/infrastructure meshes remain valid retained representations;
forest instances additionally share source geometry. This does not require every
object category to use the same mesh representation.

CPU compilation ends at `PreparedMesh`: exact packed vertex/index bytes, bounds
and shared-topology identity, accompanied by complete dependency proofs. The
ordinary world path uses one selected-tile preparation queue for ground, natural
terrain and city/infrastructure content. All configured lanes can prepare any
selected tile; fixed per-category lane reservations no longer govern this path.
Each job copies capture/projection values and river nodes, and borrows immutable
observations, coast and decoded assets under a frame lease. It never reads game
objects. Resident-handle attachment uses the separate mutable instance map.

Each lane has private ground scratch and reuses the existing private terrain
scratch sequentially for terrain and objects. Point caches and dependency
consumers reset per tile; bounded river pages survive requests and reset on world,
asset or device changes. Workers pack the tile's world-owned ranges into one
immutable GPU allocation, so uploads for independent tiles can overlap. Shared
grid indices remain shared; bed/water reuse the underlay ranges. Foreground
adoption owns validation, admission, material binding and publication, retaining
ranges without rebuilding indices or uploading these layers again. Remaining
cliff/resource/legacy adapters keep their established owners and uploads.

The queue uses the existing 64 MiB world-preparation allowance (16 MiB when that
control is disabled), counting CPU payload, GPU bytes and proofs together. Its
half-budget refill gate prevents producers from outrunning adoption. A worker
rejects a combined result above 16 MiB before GPU allocation; compiler preflights
and private two-page river caches also bound individual stages. Container growth,
packing/upload transients and active jobs are additional to ready storage and
must be checked against whole-process address-space measurements. Oversized or
failed preparation recovers through the same compilers. Cancellation, exceptions
and ordinary completion join all readers before borrowed sources or callbacks
disappear. The combined world compiler keeps its bounded worker pool alive across
frames. Ending a lease discards unstarted inputs and borrowed callbacks, while
complete owned CPU/GPU results can survive in the same reservoir. Stable
content/context keys replace camera-local slot indices. Reuse requires current
topology, coast, world and river proofs; zoom/extent are conservatively isolated.
Configuration, asset and device reset clear this owner before sources change.

Whole-world preparation also writes exact compiled values/proofs to a bounded
compressed session file (1 GiB maximum, deleted on close). The existing worker
lanes decode and validate backing after GPU eviction, then upload through the
same immutable-allocation path. This avoids geometry generation, but streaming
is still measured work and is not called resident selection. Full-map appearance
arrives in caller-thread pages of at most 128 records; the existing world builds
8×8 staggered region cores with dependency halos, including bounded wrapped edge
occurrences. It grants no additional visibility. Small maps retain the configured
GPU geometry cap; worlds above 8,192 tiles currently bound this working set to
384 MiB to preserve 32-bit address space. The supported acceptance envelope is
12,800 actual tiles. Backing is optional: missing, invalid or unavailable entries
recover through the same compiler, and reset joins readers before retiring it.

The combined preparation queue reserves a worst-case result allowance before
starting a lane, including urgent jobs, to prevent producers evicting unconsumed
results. GPU eviction uses a non-owning candidate order refreshed per cache epoch;
generation, current-frame pins and changed ages are checked at removal time.
Neither backing nor eviction candidates introduce a second render-world owner.

Current captured occurrences take priority in compilation and foreground
adoption; surrounding working-area content follows. Native occurrence/pass order
is unchanged. A completed GPU camera awaiting adoption also retains priority over
optional map and unit-pose preparation. Retained scene surfaces draw only the
pixels needed by actual view damage; speculative off-screen guard drawing is
retired because its driver submission could block new camera demand. Its padded
color/depth storage is also retired: visible circular samples and the existing
four-pixel finishing margin provide scrolling reuse, saving 224.77 MiB at
2240×1260 without changing sampling or effects. Current
demand must not acquire an unrelated speculative wait merely because its map
assembly has just completed. Whole-world geometry preparation remains independent
of the camera route.

`object_compiler.h` separates connectivity/part selection from CPU geometry.
`city_fidelity/compiler.h` retains material chunks, lighting/blockers, source
model/part identities and placements. Prepared objects preserve authored source
indices, eliminating triangle expansion followed by vertex hashing. Routes keep
their terrain-conforming strip compiler; a legal city suppresses discarded
fallback-body construction while retaining walls and constrained-site fallback.

| Object representation | Sharing boundary |
| --- | --- |
| City building bodies | Library model/part plus uniform placement; per-city lighting/blocker ownership remains separate |
| Bridges, mines, farm parts, huts/camps, fallback cities and walls | Pack-family/asset plus placement/material parameters; exact grounding and layer order |
| Routes and railroads | Connectivity-selected, subdivided terrain-conforming strips; site-specific geometry |
| City source ground and paving | Terrain-conforming vertices, atlas/coverage and ordered material chunks; site-specific geometry |
| Resources and natural vegetation | Existing animated anchors/shared natural representations |

Reduced/non-world profiles and explicit serial ground/object controls continue
through their existing scoped preparation/adoption paths, using the same
compilers. Height-bearing rigid infrastructure now uses shared immutable source
buffers and compact placement records; flat and terrain-conforming pieces retain
their grouped transformed geometry. Color, reflection and source-caster shaders
share the same placement math. Each placement has its own content version, while
split pieces preserve their original combined layer's world bounds for water
ordering and shadow/reflection context. Final occurrence/pass order remains
native capture order. Workers use only the
thread-safe D3D device for immutable allocation, never the immediate context or
native-game surfaces. Allocation boundaries remain residency/eviction boundaries;
there is no cross-tile arena, fence scheme or second presenter.

Every pixel producer, including the older screen-space block producer, borrows
all layers stored in a world owner. A terrain-only suffix is no longer a valid
ownership assumption. Shadow-page and region-image keys identify individual
immutable buffer ranges as well as revision/material, so different parts of one
owner cannot alias after culling. City scrolling has its own retained-versus-cold
integration witness, including prepared pixel blocks and source shadows.

Explicit passes name inputs, outputs and dependencies for static/dynamic geometry,
shadows, reflections, water, finishing and composition. This is not a replacement
pass order. Static color/depth can be reused only with valid contributors, lighting,
projection and receiver dependencies. A frozen unit needs no future pose work;
selected idle/work loops sample authored time, while directed action state remains
Civ III's authority. Directed-motion interpolation is a separate integration task.

The [direct unit scene contract](direct_unit_scene_contract.md) renders selected
unit geometry into a bounded transparent attachment, then uses exact native
composition above the map. This is the user's final ordering policy: units remain above tall neighboring
map objects too. Unit self-depth and pose-local shadows
remain intact; map depth/provenance and per-unit map captures are no longer inputs.
The existing pose owner reuses GPU vertices, shadow inputs and cropped body
contributions under its existing budget. Identical body samples need no second
rasterization; each occurrence still receives its own native underlay composition.
Replay collects current pose requirements before map execution so the existing CPU
workers can prepare them together. Allocations recycle after their old identity retires. Clearing, extraction and composition
follow the changed footprint while preserving exact native raster coordinates.
Separate direct/compatibility scratch avoids route-switch reallocations. Oversized
canvases use bounded resident GPU poses. No CPU body roundtrip or second presenter
is introduced; native unit/UI order, action and visibility authority remain intact.

Shoreline effects use the shared dynamic pass and damage owner with resource
bodies/shadows. Retained wave cells own immutable geometry; occurrences carry
native projection and visible/frozen time. Old and new animated footprints are
cleared and rebuilt from retained static geometry before dynamic contributions.
There is one MSAA scene color/depth set. The former static pixel-backup owner,
its tiled attachments and capture/restore traversal have been removed. Finishing
still preserves exact per-sample HDR/depth, alpha order and circular damage.

Reflections own one guarded resolved atlas and a bounded map of exact dependency
keys for its cells. Only cells intersecting water receivers and conservative
explored/visibility-feather coverage are constructed. Unchanged cells stay in
place; changed cells are copied directly from small mirror scratch. No duplicate
resolved reflection-page texture cache participates in the shared path. Keys
are withdrawn before mutation, so a failed partial update cannot certify stale
samples. Hidden atlas regions may contain old data; they have no selected reader
and must be rebuilt before becoming relevant. Water time does not invalidate
static reflection content. Camera/content/light dependencies remain explicit.

Static and water submissions retain selected occurrences, shadow-page batches
and caster inputs under the existing lifetime signatures. Dynamic changes redraw
only their damage; frozen water is redrawn when another contribution or exposure
requires it. Borrowers retire with their source assembly. Native unit/UI order,
config-off forwarding, authored quality and independent ambient clocks remain.

A shared 1 GiB logical envelope covers scene attachments, reflection atlas/scratch,
unit scratch and reproducible region/unit caches. Required targets take priority;
cache capacity uses the remainder and tightens under process VA pressure. The
scene target cap is 672 MiB, unit scratch 96 MiB, and unused unit scratch retires
after one second without actual raster use. These are allocation accounting,
not measured VRAM residency or a cap on the entire process. Geometry, source
assets and native published fronts remain separately owned and measured. The
compatibility path retains its required regional cache; the shared path gives
that superseded page cache zero capacity. There is no second presenter or clock.

Optional absent/disabled wave assets retain ordinary water and request no wave
animation. Open-water normals and connected river flow now share that dynamic
pass, with immutable authoritative inputs and visibility-frozen samples.

The tactical pass consumes native selection anchors, scoped route line/text draws
and grid-setting/tile anchors as immutable primitives. It joins the same dynamic
native composition history as unit bodies, preserving copy/erase, clipping and UI
order. Analytic GPU coverage and one generic font atlas replace FLC cursor pixels
and native red lines/turn text. Packed color conversion remains on the GPU. Input
payloads share the retained-history budget; no second scene, input handler or
pathfinder is introduced. See the [tactical contract](tactical_overlay_contract.md).

Raster caches skip useful work but do not define world ownership or force the map
into independently rebuilt mini-scenes. Choose viewport, regional or hybrid working
surfaces from pass dependencies and the combined 32-bit memory budget. A raster
cache miss should still be able to draw reusable scene content efficiently.

## Scheduling and publication

Durable scene changes are distinct from replaceable view requests. Cancelling an
obsolete camera must not lose a content update or discard useful completed work.
Workers participate in the frame pipeline: dependency-ready content updates,
view/pass selection and animation preparation may run concurrently over immutable
inputs. Current-frame work and speculative preparation are distinct scheduling
classes; current dependencies take priority. Prepare reusable content broadly and
future pixels selectively. Existing queue boundaries and worker counts may change
when this removes duplication or waits; a generic job-system rewrite is not required.
Bound queues, uploads and residency. More workers or CPU usage alone is not a win.

M3.1 connects authoritative native captures to `ScenePublication`, a bounded,
coalesced change journal inside the existing worker. Capture copies tile updates,
visibility and immutable topology/environment/time metadata before enqueueing a
view. The worker adopts accepted changes into the existing `CapturedScene`
between jobs, including when the carrying camera request was cancelled. Full
captures can remove objects; lightweight topology halos cannot. Historical view
sampling updates observations without rolling back authoritative appearance or
attaching old meshes to a newer revision. Configuration, map and viewer changes
retire the previous scope. Unit spawn/action/despawn remains in `UnitInstances`.
M3.2 extends the existing bounded camera queue to resident GPU output. Exact
copied requests share one active and one replaceable pending slot with the CPU
camera route. A changed request cancels obsolete assembly without cancelling
accepted world changes. Duplicate requests retain their ticket; pending and
superseded polls leave caller output untouched. Only explicit successful adoption
imports the completed immutable texture into the native composition session.
Adopted map tickets are distinct from request tickets. Native image operations
can pause and resume assembly while retaining the currently adopted map.

The existing synchronous `c3x_renderer_gpu_render` uses this same queue and
adoption path; the old direct GPU render branch is removed. Optional GPU
camera begin/poll exports expose the production path without changing existing
ABI layouts. Begin and pending polls do not join active rendering. Successful
polls still perform a bounded session import on the owner thread; native calls
and foreground GPU operations remain serialized. The M3.6 caller below uses
nonblocking polling only where the complete native transaction can defer; directed native work retains an exact barrier.

M3.4 adds `c3x_renderer_gpu_camera_poll_view`, an optional atomic description of
that same GPU adoption. Under the existing call/worker gates it returns the
composition image/ticket, request ticket, copied epochs and frame, ordered native
occurrences, replacement/fallback coverage and pixel phase. Device generation,
content revision, session and actual sampled clock travel with the image. It reads
the adopted owner, never a newer pending request, and writes no caller result
unless adoption succeeds. Borrowed occurrence/coverage arrays remain owned until
the next successful map adoption, CPU-map switch, configuration or reset; callers
remain serialized. No extra scene/texture copy is required by the atomic poll.

Camera ticket allocation survives worker/device recreation, so a retired request
cannot alias a new request after reset. GPU adoption retains the previous
publication owner until import succeeds; an import failure restores that owner
and leaves caller output untouched. This does not replace M3.7's device-loss and
native recovery policy. Prepared CPU/GPU views refresh clip, scheduling hints and
animation metadata together without relabeling their sampled clock. Donor and retained-scroll reuse also preserve the relative order of visible
occurrences: reordered overlapping graphics require fresh pixels even when their
content is otherwise identical. Prepared pixels require exact occurrence
coordinates as well as canonical content; wrapping does not by itself prove that
world projection, material and postprocessing inputs are interchangeable.
The native bridge below advances the complete camera/canvas transaction;
partial terrain publication alone cannot establish coherent native output.

M3.5 exposes that result through the existing native `CompositionOwner` with
`c3x_renderer_native_camera_request` / `c3x_renderer_native_camera_poll`.
Request copies inputs through the same durable journal and replaceable camera
queue. Completion posts the registered `C3X.Renderer.CameraReady.v1` thread
message to the requesting thread. Its ticket is a wake hint only; polling still
checks the current request, and a bounded retry handles failed message delivery.
No callback enters Civ III or initiates a native draw from the worker.
Polling does not wait for unfinished rendering or optional preparation;
busy publication gates return pending. A ready poll performs the existing session
import and native-image admission, then returns the exact atomic view. This ready
adoption still has a measured cost; the API does not promise a zero-duration call.
The synchronous PREPARE entry shares the same native image preparation/commit
owner and remains the explicit compatibility barrier.


M3.6 adds a caller-thread `Navigation` owner inside `CompositionOwner`. Native
`move_camera` computes/clamps/wraps manual pan intent, a capture-only traversal
copies it, and the small injected adapter restores the displayed camera before
returning. A new `Animator_update_display` inlead polls before native camera,
erase and wrap canvases advance. Pending calls still run the native Animator;
they can defer only while its own early-return predicate is true. If gameplay,
UI or an action requires work, the queued destination takes the exact path.
A scoped native tile-centering hook preserves immediate vanilla selection,
action-centering and programmatic movement, including reason-1 centering calls.
Zoom/projection changes also retain exact behavior.

Native picking, culling, selection/path anchors, grid and native labels all read
the same displayed native camera. No separate picking transform, input listener,
worker game pointer or native animation scheduler is introduced. After readiness,
a fresh complete capture must still match the queued ordered records, topology,
visibility, projection and epochs before existing PREPARE/COMMIT may consume it.
The comparison excludes only time/scheduling hints; the actual sampled clock
remains attached to the pixels. Repeated equivalent pan requests keep their work;
native-clamped no-op movement queues no capture or rendering.
One copied comparison snapshot lives in the DLL until adoption or retirement.
Retries use the existing native Animator cadence; this is not a claim that every
camera operation is nonblocking or that native input-to-display is below 66 ms.

M3.7 uses the same transaction for retirement and recovery. Failed input copies
cancel their worker ticket before returning; failed poll/import/admission advances
only the intended native camera and grants no coverage, forcing a fresh exact
render. Allocation of the caller client/adapter is atomic. Lifetime, projection
and viewer changes discard intent instead of applying it to a different scene.
Configuration-off returns an eligible queued destination before the first native
image operation or unload, even if Animator has not run yet. Reload discards it.

Reset, pack/definition reload and config-off share a checked native/display
handoff. They retire unpublished navigation and tactical capture before attempting
to flush/read back owned native images and preserve the actual displayed window.
This includes CPU-source presentation without a map composition owner. Failure
keeps the necessary DLL references, hooks and ownership alive, and marks a failed
scene unload unavailable; native map drawing
cannot resume through stale CPU storage. A successful retry completes teardown,
and only a fresh capture can publish after recreation. Recoverable allocation,
cancellation and reset are distinct from physical device removal: if GPU-only
native pixels are irretrievable, the established terminal barrier remains closed.
It cannot promise lossless recovery without rebuilding those native surfaces.

Pending grants **no coverage for the requested view**. Output and native pixels
remain unchanged; COMMIT is rejected until a successful poll and caller ownership
validation. Old pixels are not certified for the new camera. The caller must defer
that camera's entire native map transaction, not just its terrain insertion.
Unflushed prior native commands reject asynchronous admission rather than silently
flushing/waiting. A changed or cancelled ticket, retired destination or changed
surface extent cannot adopt. The owner retains only the native destination token
on its caller thread; workers receive copied scene inputs, never JGL/game pointers.
Ambient visual ticks yield to active camera work instead of cancelling and joining
it to animate an older front.

The connected native fixture exercises request/poll/validate/commit followed by
real JGL units, UI, copies and presentation. The injected bridge now polls before
Animator updates its canvases. Existing m71 PREPARE consumes validated ready
navigation, while incompatible captures and explicit native barriers render
exactly. Extracted hook tests prove the call boundary and camera decisions;
fixture success does not replace the M3.8 live-game acceptance checkpoint.

The renderer now owns independent visual scheduling through the existing worker
and presenter. It does not request Civ III redraws. Native camera, visibility,
action and UI changes still arrive through hooks/captures. General nonblocking
camera publication must advance pixels, overlays, fog and picking coherently;
stale pixels cannot be relabeled as the current view. Complete authoritative
capture remains valid input even before finer mutation notifications exist.

GPU-admitted map/pose/native composition stays resident through presentation;
explicit CPU access barriers and config-off retain their established behavior.
Do not reintroduce routine map readback. Native UI source decoding and text layout
need not become GPU algorithms to preserve resident map output. The visual clock
currently depends on the UI message pump; its 33 ms timer is an opportunity,
not a sustained-FPS guarantee. [Frame ownership](visual_frame_ownership.md)
documents the implemented lifecycle and composition details.

## Firm contracts

- Civ III owns gameplay, native screen anchors, visibility decisions, borders, labels,
  selection, UI, picking and action lifecycle. No second simulation or presenter.
- Custom-on map units are 3D; native UI portraits and config-off remain native.
  Custom-on map-plane failure must not silently replay native terrain.
- Local validity includes complete captured appearance, algorithmic neighbor and
  connectivity dependencies, contributors, assets and lifecycle revisions.
  Global invalidation remains appropriate for map replacement/global changes.
- Publication carries pixels, view/clock, coverage and replacement ownership
  together. Prefetch/caster-only content cannot authorize visible replacement.
- Account jointly for CPU address space, GPU resources, transient targets and
  in-flight ownership. Do not hide excessive total use behind independent caps.
- Preserve source normals, transforms, UVs/material channels, accepted depth/color
  differences and environment contracts. M9/M10/M11 wonders/Districts stay deferred.
- Extend working owners and replace superseded routes. Preserve executable
  behavioral tests and reproducible controls; do not maintain a parallel renderer.

## What the Firaxis material contributes

The supplied *Firaxis LORE* GDC 2011 presentation (Civ V, especially slides 8–14
and 17) supports immutable resource bundles, self-contained command streams,
job-based preparation, low allocation and redundant-state filtering. These are
useful design principles, not evidence of Civ VI's exact internal implementation
or transferable FPS. Installed Civ VI asset/ArtDef findings remain separately
identified in the [source findings](source_art_findings.md).

Retain GPU-ready descriptions as well as content: a normal frame should select
and submit cheaply. D3D11 command lists, GPU deformation/skinning, broader pose
batching and terrain synthesis are possible mechanisms, not prerequisites.
Choose them from measured costs after the connected scene path exists.

Map fog coverage consumes copied API 18 visibility in the final output pass; native
fog forwards only with custom rendering off. Dynamic input lifetime and clocks
follow [the immutable input contract](dynamic_scene_input_contract.md).
