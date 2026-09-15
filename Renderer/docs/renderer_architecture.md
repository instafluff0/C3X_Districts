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

Explicit passes name inputs, outputs and dependencies for static/dynamic geometry,
shadows, reflections, water, finishing and composition. This is not a replacement
pass order. Static color/depth can be reused only with valid contributors, lighting,
projection and receiver dependencies. A frozen unit needs no future pose work;
selected idle/work loops sample authored time, while directed action state remains
Civ III's authority. Directed-motion interpolation is a separate integration task.

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

- Civ III owns gameplay, native screen anchors, visibility, fog, borders, labels,
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
