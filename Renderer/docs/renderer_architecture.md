# Renderer architectural destination

The 2026-09-13 implementation objective in [the retained plan](retained_renderer_plan.md)
supersedes this document's single-experiment queue, tooling-first sequence and
per-increment performance acceptance rules. Build and measure the connected
representative architecture; retain the correctness, ownership, measurement,
asset and approval contracts below. Historical experiment rules do not select work.

The destination is a persistent scene with reusable world content, spatially
selected draw lists and explicit render passes, publishing an off-screen bitmap
through Civ III's existing map boundary. Camera movement should mostly change
selection, projection and composition; it should reuse unchanged scene content.
This is design guidance, not a claim that these capabilities already exist or
an instruction to rewrite the renderer.

The authorized [GPU composition implementation](gpu_composition_probe.md) now
connects the production native map owner to the retained renderer. The injected
map boundary prepares resident output, checks native replacement ownership, then
commits pixels into its admitted map canvas. One caller-owned adapter routes
native image copies, sprites, opacity transitions, lookup/FLC effects, cached text,
prepared units and final transfer through the
existing GPU worker. Native pointers/HDCs stay on the caller; only owned commands
and source pixels cross that boundary.

Startup observation certifies complete native lifetimes. GPU admission follows
map demand and copy/save dependencies; unrelated static UI can stay CPU-generated.
Escaped public pointers/DCs revoke eligibility until reinit. Paired native-word
and full-color images preserve native key/clip/fallback behavior while retaining
BGRA map color. Separate HUD alpha-sprite programs consume resident backgrounds
and write paired results through the same compositor; bounded CPU source decoding
preserves native palette/alpha validity. Positive, in-bounds native stretches preserve GDI's sampling and
bit-combination rules; keyed copies test native words while copying full-color
pixels. Palette/whole-image fills reuse the current native palette. Transparent
self-draw retains its native traversal through an explicit CPU barrier.
Indexed sprites use index-based coverage (254/255 transparent); positive source
scaling and palette-key resolution preserve audited JGL behavior. Bounded source
decoding/resampling stays on the CPU; destination composition stays on the GPU.
Native fonts render once into bounded, background-dependent glyph data; subsequent
labels blend on the GPU, preserving full map color beneath smoothed edges. Small
text-edge color differences are user-authorized; native layout remains authoritative.
CPU map/unit requests preserve the GPU image session; configuration/reset drain
native images before retiring it.

The audited Graphsy transfer follows native tooltip/cursor drawing and presents
through the existing window on its caller thread. GPU partial transfers retain
untouched displayed pixels. Explicit GDI handoff snapshots the last displayed
image before release, independently of newer working-canvas pixels; normal GPU
presentation has no CPU shadow. CPU-generated UI sources share this presenter,
including partial transfers over a full-color map. The existing final native hook
binds this presenter on configured UI demand before any map call and after scene
unload; configuration-off remains native. UI-only demand initializes the device
without map shaders. Final transfer pauses interruptible preparation while
preserving its camera ticket/input; queue activity never chooses a GDI fallback.

The production owner passes connected JGL tests, including cancelled publication,
unchanged CPU map storage, a second view and reset handoff. This is not completion
of all-GPU gameplay: unsupported text states/GDI and sprites still force CPU
barriers. Device-loss reconstruction is user-deferred; existing failure guards remain. Prepared areas now retain CPU or immutable GPU storage under the same spatial
and content-validity owner. Cold GPU demand establishes surrounding coverage;
subsequent demand selects its viewport and can adopt a background refresh. Native
adoption retains the true ambient sample clock. Speculative unit GPU jobs consume
ready CPU content only; finite unfinished predictions wait for internal completion
notification. This keeps CPU compilation off speculative GPU ownership without a
polling timer or Civ III callback. The selected unit shadow pass now removes the dominant CPU pose-raster work;
the retained plan records measured complete-request improvements and the remaining
map cost. Actual game-screen coverage remains a native integration checkpoint;
see that plan for scope and reproducible controls.

## Authority and migration

The current checkout is the implementation authority. This document defines the
intended architecture; [the execution contract](autonomous_renderer_execution.md)
governs bounded changes; [the benchmark workflow](benchmark_workflow.md) defines
measurement, tooling deliverables and validation; [the retained plan](retained_renderer_plan.md)
records current implementation status and measured scope. Historical evidence does
not create another queue. User scope, AGENTS.md and Lab ownership/visual-acceptance
rules continue to apply; the [visible scene contract](visible_scene_contract.md)
and [native presentation constraints](native_async_presentation_audit.md) preserve
the integration boundary. Proposed internal handles do not change a wire schema.

The implemented city-profile path with waves/reflections disabled keeps circular
scene color/depth and finished output at stable world-relative sample addresses.
Camera changes select exposed/static damage and ordered dynamic passes; they do
not translate the whole scene. Finishing updates affected output with exact filter
support. Persistent captured world identity is now separate from observations;
local content proofs and a ready-instance path feed separate view assembly and
spatially selected pass inputs. Full MSAA resolve remains an explicit limit.
Fresh native captures can now acquire exact-current-camera crops inside prepared
coverage and retain bounded alternative zoom views. Cold/outside-coverage demand
still uses an exact barrier. Selected animated/output refreshes keep wider static
coverage reusable, with pending output damage and honest per-publication clocks.
The retained plan records measured scope, budgets and remaining freshness limits.

Full-detail CPU surface compilation now has a bounded helper pool sharing immutable
assets and world inputs, with private river/query/layout scratch per worker.
Source mutation joins readers; locally proven results enter the existing compiled
world owner before selected pass submission. World records retain bounded borrowed handles to immutable complete tile content.
The standard 2:1 city profile compiles full-detail ground, cities, routes and
improvements in reusable coordinates; draw occurrences supply native anchors and
projection. Persistent world-cell membership belongs to that resident content,
while each view supplies current eligibility. Unsupported bases retain the
existing projected compiler and view index. The renderer thread can compile
unstarted demanded work while helpers prepare other captured instances. One GPU
owner also prepares a finite horizon of exact ambient frames after a stationary
view is confirmed; later matching calls consume immutable publications. These
mechanisms coexist with the caller-driven native camera handoff. Camera/content changes supersede unstarted other-zoom pixel work, preserving
compiled world content; stable demand can offer it again. They do not
predict game state or add redraw callbacks. See the retained plan for controls and measured
scope; helper count alone is not a speedup claim.

Unit pose helpers retain immutable mesh inputs and compile projected body vertices
plus compact selected caster records. Resident draws submit one compatible GPU
shadow pass, then material and finishing passes consume that height texture.
No CPU height/coverage images are built or uploaded on this route. The native CPU
route retains the independent rasterizer and finishing oracle. Same-size
projection, direction, phase, lighting and CPU/GPU representation participate in
content validity; owner color remains a GPU material input. Existing bounded pose
publication and future preparation remain with the same unit owner. The nearest
map bucket still has preparation priority. This removes the unit round trip;
the selected shadow-pass comparison is recorded in the retained plan. Earlier warm idle requests were
roughly 4 ms and observed native idle cadence about 15 Hz. A user-authorized GOG timer/animator
adapter now enables 33 ms selected/working-unit visual opportunities while native
advancement stays at 66 ms and map cadence remains unchanged. Enabled injection
checks pass; actual delivery cadence and coverage beyond prepared views still
require native integration validation.
Unit playback now has a bounded instance owner fed by explicit native selection:
frozen idle/fidget units generate no future pose work; selected idle and active
work loops preserve authored source duration/frames. Directed actions keep native
cursors. Resolved pose identities feed the existing content and pixel owners;
caller timestamps alone do not invalidate a pose or authorize another prediction.

## Firm principles

- **Narrow native integration.** Verified animator/map-loop hooks may improve
  delivery or remove duplicate work, but preserve Civ III's ownership and avoid
  copying its loop or maintaining competing camera/canvas state. Prefer the
  existing capture/request/response boundary when a clean split is unavailable.
  Extra visual opportunities and faster complete rendering are separate claims.
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
| Render world | Retain tile/object identity, local revisions, bounds and spatial membership across views. | `render_core/world_topology.h` owns complete terrain topology. `render_core/captured_scene.h` separates persistent canonical render appearance/revisions/bindings from bounded current observations. Identity survives camera departure and mesh eviction; unobserved appearance remains unknown and cannot authorize drawing. Complete local proofs guard compiled-content reuse. |
| Asset registry and scene compiler | Share assets; compile changed terrain, connectivity and compound object composition into reusable content. | Existing pack loaders, fidelity adapters and `CachedTileGeometry`/`ResidentContent` in `native/c3x_renderer.cpp` own complete reusable tile content for the standard full-detail city profile. Ground, city, route, improvement and natural content share local validity and survive zoom changes; regular terrain patches share canonical connectivity. Immutable tree placements refer to resident generic source meshes; local exclusions and surface dependencies remain with compiled world content. |
| View builder | Select eligible wrapped occurrences and build pass-specific draw lists. | Separate assembly of current authoritative occurrences feeds `native/render_core/geometry_draws.h`. The retained profile queries `world_pass_index.h` membership owned by immutable resident content, intersects it with current authoritative occurrences, and feeds existing ordered compatible submissions within the shadow-page limit. Residual/non-affine inputs use the existing view index. Membership never authorizes visibility; eviction releases membership and invalidates borrowed handles. |
| GPU renderer | Own the immediate D3D11 context, GPU lifetimes, uploads and pass execution. | Existing renderer worker, `draw_cached_geometry()` and `submit_geometry()`; the retained profile owns circular scene/depth and incremental finishing. Animated resources share resident meshes with separate pose/placement. Selected forests use hardware instancing in color and shadow passes, retaining the existing material order and page limit. Other object categories and reflection profiles retain their existing submissions. |
| Compositor bridge | Publish and consume complete compatible results at the native map boundary. | The injected compositor supplies native lifecycle/visibility identity to ordinary DLL publication selection; identical queued and compatible ambient work can be consumed on native demand. Prepared nearby/zoom coverage uses exact pull acquisition; general cold/outside-coverage handoff and live acceptance remain unfinished. |

The fallback regional path still traverses layer chunks for each rectangle and
includes recursive regional/reflection work. The retained profile selects explicit
static/dynamic inputs and keeps scene/output ownership across camera changes. Current guarded map
shaders use an explicit common scene-depth basis, independent of raster translation.
The user approved the measured small D24 rounding changes; existing material order,
depth writes and MSAA4 remain. Incompatible stored depth is retired on origin changes.
Regional producers and a common consumer pass the GPU contract. The retained
dynamic pass now consumes resident static color/depth without numerical rebasing.
Its measured whole-request benefit and independent replay evidence are recorded
in the retained plan; the depth contract alone is not a performance result.

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

Civ III owns render demand. Move ordinary calls toward bounded submission and
consumption of compatible ready publications at the next native render call.
Worker completion must not notify Civ III, request a redraw, or introduce a
renderer-driven presentation loop. A bounded staging ring is a candidate readback
mechanism, not proof of reduced cost or permission to queue stale views.
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
