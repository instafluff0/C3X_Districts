# Production architecture audit

Baseline source audit of the tree based on `7e6957fb` and the subsequent
frame-working-set implementation. The implemented follow-up is recorded in
[camera consolidation](camera_consolidation_results.md). This is an ownership, scheduling and critical-
path audit, not a claim of exhaustive dead-code detection or new performance
measurement. The [roadmap](retained_renderer_plan.md) remains the implementation
sequence. Existing measured evidence is in [working-set results](frame_working_set_results.md)
and [live bottleneck findings](live_bottleneck_findings.md).

## Finding

The renderer has persistent world identity, reusable GPU geometry, independent
visual time and retained native composition. It still combines those with
camera-driven preparation, frame-scoped compiler leases and synchronous native
transactions. The follow-up retires optional prepared images and their second
geometry scheduling queue, while those remaining boundaries stay explicit. The four recent changes removed particular
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
| Content: `world_preparation.h`, `c3x_renderer.cpp::render`, `render_core/content_preparation.h` | Combined ground/terrain/object jobs use persistent workers and retained results. Inputs still borrow frame-scoped sources; `WorldJoin`/`finish_lease` waits for readers before source mutation. Demanded `take` can wait. | Keep compiler and ready-content owner. Move scheduling toward durable immutable content leases; retire frame-bound queue reconfiguration where replaced. Removing waits before replacing borrowed ownership would be unsafe. |
| Near-demand priority: `WorldPreparationSchedule::prioritize` | The separate `warm_order` producer, copied inputs and cursor are deleted. Current demand reprioritizes unfinished world regions. | Keep one preparation order and existing compiler/backing owners. No fullscreen speedup is attributed to this deletion because that recording ran no neighborhood jobs. |
| Whole-world preparation: `WorldPreparationRegion`, `WorldPreparationSchedule`, `CompressedWorldStore` | One idle region producer uses the existing compiler for near and distant work. Completed regions survive camera movement; projection/lifecycle/content changes re-arm preparation. | Keep bounded immutable region leases and backing. Frame-bound compiler reconfiguration still needs measured improvement; removing its waits without replacing borrowed input is unsafe. |
| Camera images: exact `PublishedMapFrame` | `PreparedViewArea`, padded/alternate views, prospective and refresh queues, crop adoption and injected requests are deleted. Legacy optional exports decline; exact camera requests use the retained scene. | Keep atomic current/ready publication. Validate smaller/fullscreen navigation and cold oracles; image-cache removal alone is not a latency claim. |
| Old pixel caches: `begin_pixel_neighborhood`, `viewport_cache`, `start_ahead` | Screen-space pixel-block production and viewport cache lookup are excluded from the shared-scene path. Two future CPU bitmap samples remain behind compatibility eligibility; `start_ahead` excludes native presentation with nearby availability. | Do not charge dormant caches to normal GPU frame cost. Establish supported CPU/profile callers before retiring their implementations. The old scene guard-fill removal did **not** remove the separate prepared-camera-image system above. |
| Geometry representations: `CachedTileGeometry`, `ResidentContent`, `RigidSourceGpu`, natural/ground caches | Camera occurrence records and world-owned ranges coexist. Shared layers are moved into their owning entry, not necessarily copied. Forest/rigid source meshes are shared; unique/deformed terrain and city geometry remain legitimate. Lower-width projections and diagnostic controls take alternate compilation branches. | Keep representations justified by deformation/order. Audit actual duplicate bytes and repeated compilation; do not delete every structure named cache. Reuse keys still need attention: `world_preparation_key` includes tile dimensions and target width/height. Remove only view dependencies proved irrelevant to generated content. |
| Static/dynamic passes: `SceneSubmission`, `world_pass_index`, scene damage and reflection atlas | Retained ordered submissions, indexed contributors, one scene MSAA color/depth set, current keyed reflection atlas. Static backup and duplicate shared-path reflection-page textures were removed. | Keep. Broaden compatible reuse based on attribution; preserve damage, visibility feathers, source shadows, full detail and pass order. |
| Units and time: `UnitBodyRenderer`, `UnitFramePreparation`, `unit_instances.h`, `retain_visual_map`, `visual_frame` | Pose preparation and optional future unit raster work are separate from world jobs. Ambient map samples still enter general `render`. An independent cadence thread exists, but delivery skips busy call ownership and pending/active cameras. | Keep native action authority and independent visual time. Consolidate GPU priority policy across camera, ambient, unit and speculative work; make ambient sampling consume stable scene inputs without unnecessary general preparation. A running clock does not guarantee a displayed frame. |
| Native composition: `native_composition_owner.h::map`, `gpu_image_worker_client.h`, `gpu_composition_session.h`, `retained_composition.h` | Valid navigation results can be adopted; otherwise map prepare flushes and calls exact rendering synchronously. Commands batch, but create/destroy/upload/readback and frame boundaries flush. Saved native images and retained composition have real lifetimes beyond a map frame. | Keep exact native semantics, keyed/full-color surface handling and required CPU access. Consolidate compatible operations into fewer transactions and shorten exact map preparation. Do not remove genuine ownership barriers or treat all texture copies as redundant. |
| Worker/publication: `submit_locked`, `ForegroundCameraPause`, `adopt_gpu_camera_locked`, `gpu_native_presenter.h` | One GPU owner, a call gate, multiple job classes and explicit camera publication. Foreground image operations can pause/cancel active camera work. Some polling paths avoid optional-work joins; submit still waits for its command. | Keep one immediate-context owner and atomic pixels/view/coverage publication. Consolidate arbitration and measure cancellation/restart cost. An asynchronous camera request does not make the surrounding native transaction asynchronous. |
| Memory/recovery: `frame_working_set.h`, `preserve_process_headroom`, composition and geometry owners | Cache admission now subtracts live composition and simultaneous publication occupancy as well as attachments. Geometry/assets/scratch and Civ III still have other lifetimes; process VA pressure covers the whole process. | Keep existing required-allocation limits and reclaimable cache priority. Logical allocation bytes are not physical VRAM; fixed reservations do not reproduce live fragmentation. |

## Cross-boundary problems to address together

1. **Preparation is still partly a view transaction.** The consolidated regional
   producer still feeds the render entry, while compiler keys conservatively include
   projection/extent and jobs borrow sources until a lease ends. Consolidating
   queue names alone will not remove this coupling. Separate content invalidation,
   preparation and residency adoption from view assembly using existing owners.
2. **The same GPU owner serves latency-sensitive and speculative work.** Extra
   threads cannot preempt already submitted GPU work. Bound optional batches,
   centralize their priority, and distinguish waiting for required content from
   waiting behind unrelated work. Retire the displaced pause/queue machinery.
3. **Civ III ownership still creates exact synchronous boundaries.** The recorded
   camera/map stall cannot be diagnosed from geometry hit rate alone. Measure
   preparation, view/pass assembly, GPU execution, composition and caller waits
   on one critical-path timeline; overlapping worker totals cannot be added.
4. **Memory policy is partially shared.** The recent reduction was real, but it
   is not a unified process budget. Include transient and simultaneously live
   publications, backing restores and native composition when admitting work.
   A larger address space would relieve capacity, not fix repeated work or waits.
5. **Alternate paths lack a concise retirement ledger.** A compatibility branch
   needs an actual supported caller, its contract, executable coverage and a
   reason the main path cannot serve it. A historical control alone is not an
   indefinite justification for a second implementation. Preserve the user's
   controls while migrating their implementation; do not silently drop behavior.

## Measured priority and deletion requirements

Current paired replay evidence reports map-prepare p95 around 350 ms and cold map
work around 7.5 seconds. Ambient request p95 improved to roughly 25–27 ms. These
are replay measurements, not proof of live FPS. They prioritize the complete
map/composition path; they do not establish neighborhood prefetch as the dominant
stall. The failed live tail and contiguous-address-space recovery remain open.

Within M3.8, use the existing recording first to attribute the long map boundary
and shared-worker occupancy. In that work, consolidate content scheduling and
its lifetime rules, then replace prepared-image navigation as cheap scene views
cover its supported cases. Expand accounting/admission across those same owners.
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
effects. Remaining frame-bound leases, exact native transactions and live heap
behavior are still open; neither document claims every cache unnecessary.
