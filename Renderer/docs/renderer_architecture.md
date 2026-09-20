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
receiver-shadow pages fit; draw parameters upload in bounded batches. Color,
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
and ordinary completion join all readers before the frame's borrowed sources or
callbacks disappear; queued results do not survive the frame.

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
compilers. The current object representation remains retained transformed meshes;
source-mesh instancing is a later option where it removes measured work. Final
occurrence/pass order remains native capture order. Workers use only the
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
composition above the map. This is the user's final ordering policy, including
units behind tall neighboring map objects. Unit self-depth and pose-local shadows
remain intact; map depth/provenance and per-unit map captures are no longer inputs.
The existing pose owner reuses GPU vertices and shadow inputs, recycling matching
allocations after their old identity retires. Clearing, extraction and composition
follow the changed footprint while preserving exact native raster coordinates.
Separate direct/compatibility scratch avoids route-switch reallocations. Oversized
canvases use bounded resident GPU poses. No CPU body roundtrip or second presenter
is introduced; native unit/UI order, action and visibility authority remain intact.

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
