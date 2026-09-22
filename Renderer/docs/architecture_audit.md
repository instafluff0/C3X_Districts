# Production architecture audit

Baseline source audit of the tree based on `7e6957fb` and the subsequent
frame-working-set implementation. The implemented follow-ups are recorded in
[camera consolidation](camera_consolidation_results.md) and
[durable content and transactions](durable_content_results.md). This is an ownership, scheduling and critical-
path audit, not a claim of exhaustive dead-code detection or new performance
measurement. The [roadmap](retained_renderer_plan.md) remains the implementation
sequence. Existing measured evidence is in [working-set results](frame_working_set_results.md)
and [live bottleneck findings](live_bottleneck_findings.md).

## Finding

The renderer has persistent world identity, reusable GPU geometry, independent
visual time and retained native composition. It still combines those with
camera-driven preparation, frame-scoped compiler leases and synchronous native
transactions. The follow-up retires optional prepared images and their second
geometry scheduling queue, while the later durable-content work removes combined frame leases and fuses resource handoffs. Other boundaries stay explicit. The four recent changes removed particular
duplicate allocations and repeated submission work; they did not finish the
architectural consolidation. Describing them as removal of all redundant old
machinery would be incorrect.

The desired boundary is copied authoritative changes → prepared content → view
selection → bounded GPU execution → coherent native composition/publication.
Image reuse may accelerate this path but must not be what makes a destination
usable. Every migration must name the old producer, queue, storage and consumer
that it replaces, or document the remaining caller and retirement condition.

## Production reachability and ownership

Paths below are relative to `Renderer/native/` except `injected_code.c` at the
repository root. Names identify source symbols, not an assertion that every
branch ran in the recorded workload. Normal configuration selects city fidelity,
the shared scene surface when its extent is supported, retained world content,
and world preparation. Explicit controls, reduced projections, native CPU
ownership and unsupported extents retain alternate paths.

| Responsibility / source evidence | What actually remains | Disposition |
| --- | --- | --- |
| Authority: `injected_code.c::prepare_custom_renderer_frame`, `RendererWorker::capture_world_tick`, `render_core/world_input_capture.h` | Visible captures plus repeating caller-thread pages of at most 128 world records. The cursor wraps and the 33 ms timer skips busy ownership. This is not a complete mutation-only subscription. | Keep copying and lifecycle validation. Consolidate dirty publication only after native mutation coverage is proved; retain bounded reconciliation for gaps. Never give workers game pointers. |
| Publication: `render_core/scene_publication.h`, `captured_scene.h`, `resident_content.h` | One coalesced change journal, persistent authoritative records, and generation-checked non-owning GPU handles. Snapshot topology can be shared; captured records are compared and normalized. | Keep. These are different lifetimes, not duplicate render worlds. Measure repeated capture/comparison separately from compilation. |
| Content: `world_preparation.h`, `c3x_renderer.cpp::render`, `render_core/content_preparation.h` | Combined ground/terrain/object jobs use persistent workers and retained results. Combined jobs now own observations and share immutable coast/topology snapshots; camera changes replace pending demand without joining active producers. Demanded `take` can still wait for cold content, and can detach when its camera is superseded. | Keep the existing bounded compiler/ready owner. Frame-local combined callbacks and joins are retired. Asset/device retirement still joins; alternate worker-off/oversized paths retain their documented scoped leases. |
| Near-demand priority: `WorldPreparationSchedule::prioritize` | The separate `warm_order` producer, copied inputs and cursor are deleted. Current demand reprioritizes unfinished world regions. | Keep one preparation order and existing compiler/backing owners. No fullscreen speedup is attributed to this deletion because that recording ran no neighborhood jobs. |
| Whole-world preparation: `WorldPreparationRegion`, `WorldPreparationSchedule`, `CompressedWorldStore` | One idle region producer uses the existing compiler for near and distant work. Completed regions survive camera movement; projection/lifecycle/content changes re-arm preparation. | Keep bounded immutable region leases and backing. Combined compiler inputs now survive region/view replacement. Regional selection still enters the render owner; this is not a second compiler queue. |
| Camera images: exact `PublishedMapFrame` | `PreparedViewArea`, padded/alternate views, prospective and refresh queues, crop adoption and injected requests are deleted. Legacy optional exports decline; exact camera requests use the retained scene. | Keep atomic current/ready publication. Validate smaller/fullscreen navigation and cold oracles; image-cache removal alone is not a latency claim. |
| Old pixel caches: `begin_pixel_neighborhood`, `viewport_cache`, `start_ahead` | Screen-space pixel-block production and viewport cache lookup are excluded from the shared-scene path. Two future CPU bitmap samples remain behind compatibility eligibility; `start_ahead` excludes native presentation with nearby availability. | Do not charge dormant caches to normal GPU frame cost. Establish supported CPU/profile callers before retiring their implementations. The old scene guard-fill removal did **not** remove the separate prepared-camera-image system above. |
| Geometry representations: `CachedTileGeometry`, `ResidentContent`, `RigidSourceGpu`, natural/ground caches | Camera occurrence records and world-owned ranges coexist. Shared layers are moved into their owning entry, not necessarily copied. Forest/rigid source meshes are shared; unique/deformed terrain and city geometry remain legitimate. Lower-width projections and diagnostic controls take alternate compilation branches. | Keep representations justified by deformation/order. Audit actual duplicate bytes and repeated compilation; do not delete every structure named cache. Reuse keys still need attention: `world_preparation_key` includes tile dimensions and target width/height. Remove only view dependencies proved irrelevant to generated content. |
| Static/dynamic passes: `SceneSubmission`, `world_pass_index`, scene damage and reflection atlas | Retained ordered submissions, indexed contributors, one scene MSAA color/depth set, current keyed reflection atlas. Static backup and duplicate shared-path reflection-page textures were removed. | Keep. Broaden compatible reuse based on attribution; preserve damage, visibility feathers, source shadows, full detail and pass order. |
| Units and time: `UnitBodyRenderer`, `UnitFramePreparation`, `unit_instances.h`, `retain_visual_map`, `visual_frame` | Pose preparation and optional future unit raster work are separate from world jobs. Ambient map samples still enter general `render`. An independent cadence thread exists, but delivery skips busy call ownership and pending/active cameras. | Keep native action authority and independent visual time. Consolidate GPU priority policy across camera, ambient, unit and speculative work; make ambient sampling consume stable scene inputs without unnecessary general preparation. A running clock does not guarantee a displayed frame. |
| Native composition: `native_composition_owner.h::map`, `gpu_image_worker_client.h`, `gpu_composition_session.h`, `retained_composition.h` | Valid navigation results can be adopted; otherwise map prepare flushes and calls exact rendering synchronously. Draw batches now accompany the following create/destroy/upload/readback in one ordered packet. Frame boundaries still flush. Saved native images and retained composition have real lifetimes beyond a map frame. | Keep exact native semantics, keyed/full-color surface handling and required CPU access. The resource-operation prelude removes a separate handoff. Continue shortening exact map preparation. Do not remove genuine ownership barriers or treat all texture copies as redundant. |
| Worker/publication: `submit_locked`, `ForegroundCameraPause`, `adopt_gpu_camera_locked`, `gpu_native_presenter.h` | One GPU owner, a call gate, multiple job classes and explicit camera publication. Foreground image operations can pause/cancel active camera work. Some polling paths avoid optional-work joins; submit still waits for its command. | Keep one immediate-context owner and atomic pixels/view/coverage publication. Consolidate arbitration and measure cancellation/restart cost. An asynchronous camera request does not make the surrounding native transaction asynchronous. |
| Memory/recovery: `frame_working_set.h`, `preserve_process_headroom`, composition and geometry owners | Cache admission now subtracts live composition and simultaneous publication occupancy as well as attachments. Measured process VA now also coordinates geometry growth and compiler capacity, with bounded shared input snapshots and explicit scratch reserves. Assets and native composition remain in that measured process footprint. | Keep existing required-allocation limits and reclaimable cache priority. Logical allocation bytes are not physical VRAM; fixed reservations do not reproduce live fragmentation. |

## Cross-boundary problems to address together

1. **Preparation is still partly a view transaction.** The consolidated regional
   producer still feeds the render entry, while compiler keys conservatively include
   projection/extent. Combined jobs now own their sources; compatibility compilers
   retain scoped leases. Continue separating regional selection/adoption from
   view assembly using the existing owners.
2. **The same GPU owner serves latency-sensitive and speculative work.** Extra
   threads cannot preempt already submitted GPU work. Bound optional batches,
   centralize their priority, and distinguish waiting for required content from
   waiting behind unrelated work. Retire the displaced pause/queue machinery.
3. **Civ III ownership still creates exact synchronous boundaries.** The recorded
   camera/map stall cannot be diagnosed from geometry hit rate alone. Measure
   preparation, view/pass assembly, GPU execution, composition and caller waits
   on one critical-path timeline; overlapping worker totals cannot be added.
4. **Memory policy is partially shared.** The recent reduction was real, but it
   is not a unified process budget. The durable-content follow-up coordinates geometry/preparation using measured
   process VA, with owned-input bounds and scratch reserves. Required attachments
   and fragmentation still need whole-workload evidence.
   A larger address space would relieve capacity, not fix repeated work or waits.
5. **Alternate paths lack a concise retirement ledger.** A compatibility branch
   needs an actual supported caller, its contract, executable coverage and a
   reason the main path cannot serve it. A historical control alone is not an
   indefinite justification for a second implementation. Preserve the user's
   controls while migrating their implementation; do not silently drop behavior.

## Measured priority and deletion requirements

Current durable-content paired replay evidence reports map-prepare p95 of
333–379 ms and cold map work of 6.9–7.3 seconds. Ambient request p95 ranges from
25–37 ms. These
are replay measurements, not proof of live FPS. They prioritize the complete
map/composition path; they do not establish neighborhood prefetch as the dominant
stall. The failed live tail and contiguous-address-space recovery remain open.

Within M3.8, use the existing recording first to attribute the long map boundary
and shared-worker occupancy. The implemented follow-ups consolidate content lifetimes and replace prepared-image
navigation with exact scene views. Continue shortening view/pass assembly, GPU
execution and native composition; extend admission only from measured ownership.
Keep each behavioral change independently comparable even when implemented in
one sustained work session; do not combine unmeasured deletions into a speed claim.

Every migrated responsibility must record:

- Its surviving owner and removed queue/cache/producer/callers, including injected
  wrappers and obsolete tests where applicable.
- Any retained alternate path's actual supported trigger and retirement condition.
- Matching output/lifecycle behavior on the recorded prefix, plus cold/warm and
  smaller/fullscreen navigation, wrapping, zoom, selection/action centering,
  reveal/fog, ambient continuity, native CPU access, reset and configuration-off.
- Complete workload timings, residency/restoration/compile counts, discarded work,
  wait attribution, uploads/readbacks and peak simultaneous memory. Keep waves and
  reflections on. Reuse existing artifacts; avoid full-frame dumps by default.

This baseline audit motivated the camera consolidation. The linked results name
retired owners, retained compatibility contracts, rejected changes and measured
effects. Combined frame-bound leases are now retired. Compatibility leases, cold-content
waits, required exact native transactions and live heap behavior remain explicit;
none of these documents claims every cache unnecessary.
