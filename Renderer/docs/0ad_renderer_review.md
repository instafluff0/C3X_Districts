# 0 A.D. renderer source review

Reviewed 2026-09-12 against upstream commit
`c3ace13b54f1d8a56557a6136814d1a6ceb66779`, resolved from the requested
[0 A.D. repository](https://gitea.wildfiregames.com/0ad/0ad).
This was selective inspection of renderer, graphics and simulation/render-bridge
source, not a build, benchmark or review of the complete engine. Source links
below pin that revision. No upstream code or assets were imported into C3X.

**Conclusion:** useful as a concrete architecture reference. Its strongest
transferable mechanisms are persistent shared mesh data, dirty terrain patches,
material/pass selection and preparation shared across rendering passes. These
support our existing [architectural destination](renderer_architecture.md).
They do not establish a speedup in our renderer or remove our native bitmap
publication constraints. The [retained plan](retained_renderer_plan.md) remains
the only active task queue; this review is supporting evidence.

## Observed mechanisms and their transfer

| Mechanism | Observed upstream implementation | Implication for C3X |
| --- | --- | --- |
| Shared immutable meshes | `InstancingModelRenderer::CreateModelData` attaches GPU render data to a shared model definition once. Static model update/upload methods do no per-frame geometry work. | Separate reusable asset buffers from instance placement and visible wrapped occurrences. Camera changes should change projection/selection, not rebuild the asset. |
| Local terrain invalidation | `CTerrain::MakeDirty` maps an affected tile range to patches. `TerrainRenderer::Submit` reuses patch render data; `CPatchRData::Update` rebuilds only when dirty. | Extend current retained world geometry and dependency observations. Rebuild terrain/connectivity for semantic edits and actual dependency reach, not camera movement. |
| Material and pass grouping | `ModelRenderer` groups by shader effect/defines, sorts compatible opaque models by mesh/texture/uniform state, and reuses bindings. `ShaderManager` caches programs and effects by name plus defines. | Explicit material variants and prepared compatible draws are established mechanisms. Preserve transparent/decal/depth ordering; fewer binds alone do not prove useful batching. |
| Shared preparation across passes | `SceneRenderer::PrepareModels` gathers dirty skinned submissions from cull groups, clears each update flag and updates the unique set once. Render data survives per-frame submission lists. | Retain content/pose ownership independently of view/pass lists. Prepare a changed pose once and let dependent passes borrow it. |
| Early eligibility checks | `CCmpUnitRenderer::RenderSubmit` checks visibility and swept-sphere bounds before updating transforms and testing exact world bounds. Transform updates are deduplicated across cull groups. | Select eligible contributors before expensive per-vertex preparation where conservative bounds are available. Preserve captured ownership, wrapping, overhang and shadow reach. |
| GPU animation | `GPUSkinnedModelRenderer` shares source model buffers, retains per-model output storage, uploads bone matrices and dispatches compute skinning. | Shared geometry plus small pose parameters is a coherent eventual animation representation. It is not automatically the largest current saving, and does not authorize changing the native unit/action bridge. |
| Explicit buffer ownership | `VertexBufferManager` suballocates compatible buffers, returns releasable handles and reports reserved/allocated bytes. | Shared allocations need one owner and accounting. Our combined 32-bit address-space and transient limits remain stricter requirements; the inspected allocator does not establish a comparable hard joint budget. |

Primary source locations:

- [Static shared model data and per-model drawing](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/renderer/InstancingModelRenderer.cpp#L207).
- [Terrain dirty ranges](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/graphics/Terrain.cpp#L756),
  [retained patch submission](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/renderer/TerrainRenderer.cpp#L181),
  [dirty patch rebuild](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/renderer/PatchRData.cpp#L823).
- [Material grouping and binding reuse](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/renderer/ModelRenderer.cpp#L360),
  [shader variant caching](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/graphics/ShaderManager.cpp#L68).
- [Unique model preparation across passes](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/renderer/SceneRenderer.cpp#L189),
  [visibility and transform selection](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/simulation2/components/CCmpUnitRenderer.cpp#L391).
- [Persistent GPU skinning resources and updates](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/renderer/GPUSkinnedModelRenderer.cpp#L339),
  [buffer allocation and release](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/renderer/VertexBufferManager.cpp#L108).

## Avoid overstating the comparison

- The inspected `InstancingModelRenderer` shares mesh storage but calls
  `DrawIndexedInRange` for each model. Its name is not evidence of hardware
  instancing that combines many objects into one draw.
- Unit submission still scans the unit array; a coarse spatial structure is an
  explicit TODO. `Interpolate` also updates every actor's animation, with a TODO
  about off-screen updates and sound semantics. The source does not justify
  claiming that 0 A.D. already eliminates all off-screen animation work.
- Dirty terrain patches rebuild their vertices, sides, indices, blends and water
  together. Finer invalidation within a patch is an explicit TODO.
- 0 A.D.'s ordinary [frame path](https://gitea.wildfiregames.com/0ad/0ad/src/commit/c3ace13b54f1d8a56557a6136814d1a6ceb66779/source/renderer/Renderer.cpp#L520)
  owns a swap chain and presents it. Synchronous framebuffer readback appears in
  its screenshot paths. C3X must publish a completed bitmap through Civ III's map
  boundary with current anchors and native overlays. Direct presentation, camera
  ownership and an engine-wide scene simulator do not transfer to that bridge.

## What this changes in our priorities

The durable priority is reusable world content: shared meshes/materials, stable
instances, local revisions and selected pass inputs. Current `CachedTileGeometry`,
`NaturalTile` world meshes, dependency records and shared natural GPU buffers are
real footholds. They are not yet a complete persistent appearance/instance model.
Avoid replacing them wholesale or describing the renderer as starting from zero.

Our recent texture-array candidate grouped guarded screen regions. It did not
create the shared world-asset representation illustrated by 0 A.D. Its matched
dense navigation result failed, so the implementation was removed. Preparation-
only binding changes also failed to establish a useful whole-request benefit.
Those outcomes close those hypotheses; they are not architectural milestones.

A low-cost applicable mechanism remains **material shader specialization**.
`compose_resource_animations` constructs bodies with material 21 and shadows with
panel 1/surface kind 15, but `submit_prepared_resource_region` still selects the
general feature and terrain pixel shaders. The source adapter can compile variants
with those known inputs while retaining the existing shading formulas, alpha
coverage, depth, projection, guarded surfaces and publication. Our inference is
that eliminating unrelated shader paths may reduce GPU work. Upstream variant
caching supports the design pattern, not the magnitude or existence of a saving.

This is a bounded application of explicit material/pass responsibilities, not
completion of persistent world instances. The existing profiling-off dense trace
attributes about 114 ms of a roughly 194 ms request to animation composition,
including GPU completion. That entire phase is only an upper bound: shader work
is not separately isolated, and reconstruction, copies and pose work remain.
Use exact full-redraw checks and matched complete-request timings to decide.
Instruction counts, shader size or lower submission counts cannot pass the gate.

GPU skinning and earlier culling are credible architectural mechanisms, but CPU
pose/selection work in the existing traces is only roughly 12–16 ms per request.
Eliminating that component alone cannot meet the current 20 ms usefulness
criterion. A broad animation rewrite solely to chase that component is therefore
not the immediate low-hanging fruit. Likewise, another selection-only index needs
a different mechanism from the previous index that reduced checks but slowed the
complete workload.

The dense 100x100 world fixture already exercises incremental navigation with
realistic synthetic object density. It is not a captured live full-game session.
Below-100 ms navigation, sustained busy/effect/lifecycle workloads and native
presentation remain unpassed gates. Reviewing 0 A.D. changes implementation
judgment; it does not waive any of those gates or authorize staging or launch.
