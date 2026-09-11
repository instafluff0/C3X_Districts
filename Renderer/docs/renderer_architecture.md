# Renderer architectural destination

The destination is a persistent scene with reusable world content, spatially
selected draw lists and explicit render passes, publishing an off-screen bitmap
through Civ III's existing map boundary. Camera movement should mostly change
selection, projection and composition; it should reuse unchanged scene content.
This is design guidance, not a claim that these capabilities already exist or
an instruction to rewrite the renderer.

## Authority and migration

The current checkout is the implementation authority. This document defines the
intended architecture; [the execution contract](autonomous_renderer_execution.md)
governs bounded changes; [the benchmark workflow](benchmark_workflow.md) defines
measurement, tooling deliverables and validation; [the retained plan](retained_renderer_plan.md)
alone records current status and the single next task. Historical evidence does
not create another queue. User scope, AGENTS.md and Lab ownership/visual-acceptance
rules continue to apply; the [visible scene contract](visible_scene_contract.md)
and [native presentation constraints](native_async_presentation_audit.md) preserve
the integration boundary. Proposed internal handles do not change a wire schema.

Translating a valid static front, composing exposed strips and dirty route/object
bounds, and updating ambient/unit layers independently is the **current migration
step**. It remains the selected near-term path and workload gate, not the permanent
definition of rendering. Retained scene resources should also make a raster miss
or a new view efficient. A changed implementation decision needs causal evidence
and an update to the single active task in the retained plan, not a parallel roadmap.

## Firm principles

- **Authoritative C3X ownership.** C3X owns game state, visibility, anchors, camera,
  time, seasons and action lifecycle. The renderer consumes immutable copied
  values, never game pointers or native canvases. Fog, borders, labels, selection,
  unit HUD, UI and picking keep their native owners. There is no second presenter,
  camera or game loop. Config-off preserves native rendering; custom-on map-plane
  failure must not replay native terrain. Preserve the separate unit fallback.
- **Persistent scene identity.** A world record survives a change of view. Separate
  an asset (mesh/material/animation), a world instance (identity, placement and
  state), and a visible occurrence (instance plus wrapped offset and captured
  projection). Multiple wrapped occurrences share resources without conflating
  their anchors, visibility or replacement ownership. Removal/reuse must not let
  an old reference resolve to a new object. Variation uses stable identity and
  named seeds, not traversal or cache insertion order.
- **Local invalidation.** Rebuild content when its actual dependencies change.
  Camera motion does not reseed placement; animation time does not invalidate
  static vertices. Dependency reach follows the algorithm, including connectivity,
  coast queries, overhangs, shadows and reflections. Global generations remain
  necessary for map replacement and truly global changes; local edits should not
  invalidate unrelated content. Broad keys may only be removed once replacement
  dependency tracking is complete and tested.
- **Bounded resource ownership.** Every allocation, queued job and publication has
  an owner and lifetime. Account jointly for CPU address space, GPU residency and
  transient targets, count shared allocations once, and protect in-flight data.
  Device reset invalidates GPU objects while preserving reusable CPU descriptions
  where valid. Independent cache caps must not hide an excessive combined footprint
  in the 32-bit game; retain the workflow's headroom and capacity gates.
- **Exact publication.** Pixels, coverage, replacement flags, occurrence order,
  camera/projection, visibility and lifecycle identity travel together. Cached
  or prepared geometry does not authorize display of uncaptured content. Caster-
  and prefetch-only records cannot acquire map replacement ownership. Reject
  incompatible completed work rather than relabeling it for the current view.

## Intended responsibilities and current footholds

These are logical responsibilities, not mandatory class names, directory splits
or an ECS framework. Extend existing owners incrementally; avoid a second scene
simulator or duplicate category implementations.

| Responsibility | Intended ownership | Current foothold |
| --- | --- | --- |
| Game scene bridge | Capture authoritative values and lifecycle changes; publish immutable input. | Capture and composition in `injected_code.c`, versioned API in `C3X.h` and the native DLL. |
| Render world | Retain tile/object identity, local revisions, bounds and spatial membership across views. | `render_core/world_topology.h`, captured records and dependency observations; a complete persistent appearance mirror is still a destination. |
| Asset registry and scene compiler | Share assets; compile changed terrain, connectivity and compound object composition into reusable content. | Existing pack loaders, fidelity adapters, `CachedTileGeometry` and shared natural world meshes in `native/c3x_renderer.cpp`. |
| View builder | Select eligible wrapped occurrences and build pass-specific draw lists. | Existing contributor/bounds queries and assembled geometry layers; general persistent selection remains to be developed. |
| GPU renderer | Own the immediate D3D11 context, GPU lifetimes, uploads and pass execution. | Existing renderer worker, `draw_cached_geometry()` and `submit_geometry()`. |
| Compositor bridge | Publish and consume complete compatible results at the native map boundary. | DLL publication records and injected synchronous composition; ordinary native asynchronous coordination is unfinished. |

The current draw path still traverses layer chunks for each rectangle, and pass
submission includes recursive regional/reflection work. Those are migration
footholds, not requirements to preserve the current function structure. Some
retained capabilities are opt-in; source presence is not proof of live-game use.

## Preferred mechanisms

### Persistent spatial selection

Use the existing canonical tile parity lattice for compact tile state. Stable
handles with generations are a suitable mechanism for sparse object records.
A coarse world grid or chunks should retain membership, revisions and reusable
geometry/instance references, with moving membership updated independently.
Keep one content owner but index every cell intersected by the relevant bounds;
deduplicate query results. An anchor tile alone cannot describe an overhanging
mountain or a caster/reflection influence. Main, shadow and reflection queries
need their respective bounds and exact eligibility checks.

World chunks own reusable content; raster blocks cache projected pixels. Their
sizes, identities and lifetimes are independent. A persistent index should reduce
selection **and actual submission work**, including repeated layer scans, rather
than only accelerate cache-key construction. The previous candidate index cut
bounds tests by 98% but regressed frame time, as recorded in
[the benchmark workflow](benchmark_workflow.md#preserve-the-investment).
Reopening that hypothesis requires a materially different mechanism and a reason
it can reduce total elapsed cost, including maintenance, query and batching costs.

### Content representation and batching

| Content | Preferred reusable representation |
| --- | --- |
| Ground, transitions and riverbeds | Regional meshes with local terrain/topology dependencies. |
| Roads, rails and improvements | Retained connectivity compiled into regional geometry or reusable segments, including dependent joins. |
| Repeated vegetation, rocks and resources | Shared source meshes/materials and compact instance records where compatible. |
| Terrain-conforming relief | Shared source assets plus retained deformation/transforms for the actual terrain dependencies. |
| Cities and walls | Persistent component selections and transforms; growth/style/state changes update composition. |
| Water and animated resources | Retained geometry plus independent time/pose parameters and explicit scene/depth dependencies. |
| Units | Shared bodies, materials and clips with stable per-unit/action state; preserve the current body-image service during a separate bridge migration. |

Separate category placement semantics, material/depth/blend behavior and pass
membership. Prefer shared meshes and instancing for compatible repeated content,
and bounded regional vertex/index batches for unique geometry. Keep subranges and
bounds for local rebuilds and culling. Select candidates, check exact eligibility,
then group compatible draws and batch parameter uploads. Reordering must preserve
transparent, cutout, decal, water and painter/depth contracts; matching materials
alone do not authorize regrouping. Source normals, transforms, UVs and material
channels keep their existing preservation rules.

Units follow C3X's action director, timing, facing, stack order, interruptions and
cleanup. Batched pose requests, unit-plane rendering or GPU skinning are possible
later mechanisms, not permission to synchronize independent actions or bypass
the compatibility service. Wonders and Districts remain deferred; shared machinery
must preserve their existing contracts without starting their implementation.

### Dependencies and caches

| Retained data | Identity and validity inputs |
| --- | --- |
| Imported asset | Content and import version. |
| Compiled world content | Stable instance/region identity, semantic state, relevant neighbors and asset versions. |
| GPU allocation | Compiled resource identity and device generation. |
| Prepared draws | Instance membership, transforms, bindings and pass eligibility. |
| Shadow/reflection results | Actual contributors, lighting and relevant view/receiver dependencies. |
| Raster region | All contributors, projection, visibility, lighting, effects and output settings. |
| Completed publication | Exact capture/view/lifecycle identity and matching ownership metadata. |

Ordinary world-mesh identity should exclude camera position. Color and time should
be material/instance parameters where they do not alter geometry. A road edit
invalidates dependent joins; forest removal invalidates membership and affected
shadows; city growth invalidates that composition and dependent nearby exclusions
or lighting. Visibility changes alter eligibility and composed images; lighting
changes alter shading/shadows. Neither inherently requires rebuilding static meshes.

Static geometry does not guarantee static pixels. Bitmap reuse remains valuable
when it saves rendering/readback, but must retain the depth and scene information
required by dependent effects. Cache policy must leave misses able to draw resident
scene content efficiently. Budget visible work first; speculative preparation must
not displace the working set needed for current interaction. Distant preparation
needs complete captured appearance, not topology-only guesses.

### Explicit passes and scheduling

Give shadow, reflection, main geometry, overlays, blended/water, animation and
finishing/output work explicit selected inputs, outputs and dependencies. Preserve
the verified ordering, projection, guards, depth and color transfer; this list is
not a replacement pass order. Shared passes should reveal why content is resubmitted
or an effect invalidated without accumulating category-specific orchestration.

Keep immediate-context work under the existing single GPU owner. CPU loading and
compilation can prepare immutable results. Distinguish durable ordered scene changes
from replaceable camera requests: cancellation may supersede a view, but must not
lose world changes or discard useful completed preparation. Bound uploads, pending
work and resource residency. Prefer visible missing content over speculation.

Move ordinary redraws toward bounded submission, polling and publication only with
a concrete native completion/redraw path. A bounded staging ring is a candidate
readback mechanism, not proof of reduced cost or permission to queue stale views.
Current native overlays and picking cannot advance against an incompatible old
bitmap. Worker throughput alone does not establish native presented-frame cadence;
follow the native presentation constraints and require live evidence at that gate.

## Choices requiring evidence

Chunk dimensions (including proposed 8x8 or 16x16 valid-tile groups), category
coverage for instancing, batch boundaries, raster-cache policy, staging-buffer
count and migration order are hypotheses. They are not immutable architecture
requirements or additional active tasks. Choose a bounded mechanism from the
measured dominant cost and explain what work it eliminates. A structural capability
and its performance effect must be reported separately; fewer checks, more cache
hits or fewer draws do not alone establish a faster realistic session.

## Validation and bounded completion

Tests protect observable contracts: pixels, depth, visibility, ownership, anchors,
wrapping, action timing, cancellation/recovery and memory bounds. Structural tests
may change with an intentional implementation change when equivalent behavioral
coverage remains; preserving today's cache layout or function names is not the
goal. Never weaken correctness to obtain a timing pass. Unchanged-appearance
optimizations require independent full-redraw parity; deliberate appearance changes
follow Lab review.

Architecture witnesses should show bounded consequences: a pan reuses unchanged
world content; a road edit changes its dependency neighborhood; city growth updates
its composition; animation advances dynamic state; removal, wrap and reset cannot
revive stale identity or ownership. Add only witnesses needed for the selected
change, through the existing harness. The workflow advances useful exact results
from tiny fixtures to dense resident movement, full busy/cold/evicted/lifecycle
pressure and native presentation. It supplies the timings and stop conditions.

The tooling phase ends at the workflow's three reusable deliverables and their
specified validation. Record completion in the retained plan and move its single
next task to the scene/rendering capability selected by that evidence. Further
harness work requires a specific missing measurement or correctness check that
can change the implementation decision. This document creates no open-ended
tooling program or rewrite mandate.
