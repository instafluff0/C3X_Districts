# Native asynchronous presentation audit

Current implementation updated September 14, 2026. The GOG camera inlead was
added with explicit user authorization; other addresses remain `0x0`. Candidate
compilation/replay does not install hooks or launch Civ III.

The current-camera correction also removes the remaining m71 camera swapping
and unused requested-camera state. Native Animator computes canvas/wrap copies
before m71; polling cannot replace that camera. The executable fixture now models
that ordering and rejects completed old-camera/old-projection tickets, including
camera changes bypassing the movement hook. General nonblocking scrolling is
still unfinished; the game remains the sole caller and camera owner.

A separate unit visual-cadence candidate is implemented but disabled pending the
five symbols in [the patch ledger](civ3_patch_dependency_ledger.md). It reuses the
native timer/animator, preserves native advancement and keeps map cadence unchanged.
The renderer does not notify the game. See [the retained plan](retained_renderer_plan.md)
for the verified ABI, lifecycle tests and rejected global map-rate measurement.

## Live checkpoint correction

The default-enabled bridge failed actual scrolling: requests changed but displayed
camera coordinates did not advance, and old-view exact redraws repeatedly cancelled
camera work. Native movement now keeps its immediate camera/bounds changes and is
an exact-render barrier. Stationary publication and CPU preparation remain enabled.
The former native fixture omitted animator/canvas camera state outside m71; its
success did not certify the full integration. The exact publication-proof mismatch
is unresolved. See the retained plan for trace counts and recovery validation.

The following describes the prior bridge, retained as implementation evidence;
its relative-input holding behavior no longer runs in the native movement hook.

## Implemented caller-driven displayed view

The bridge now defaults on for the user-authorized `INSTALL.bat` workflow, with
no environment settings required (`C3X_RENDERER_NATIVE_ASYNC=0` is retained only
as a diagnostic opt-out). It connects the existing captured
world, camera queue and immutable publication at m71/m19. Native camera fields and
bounds describe the displayed view between calls, including native overlay
placement, direct inverse picking and unit culling. The new `move_camera` inlead
accumulates relative scroll intent separately and delegates wrap/clamp to the
original function. Programmatic recentering, animation takeover, zoom/resize and
invalid display proofs use an exact-render barrier. Complete topology/visibility
capture and all optional exports are required; otherwise the exact path remains.

On each native draw, the bridge polls its one active ticket, selects the candidate
native camera, then recaptures that view before acquiring pixels. The new optional
`camera_present_view` validates exact ordered appearance, topology revision,
visibility and native lifecycle epochs. Only time and dirty scheduling hints may
differ. The lease exposes the actual displayed clock. Failed validation requires
exact full-detail rendering; failed composition clears the map plane in this mode.
No low-detail preview or changed native suppression ownership is introduced.

After native map traversal, a capture-only call through the existing m21 vtable
builds the latest requested view using the same m19/frame owner. It queues only
when the active slot is free. New relative scroll intent can accumulate without
cancelling that request on every draw. Native fields then return to the displayed
view. Completed work stays in the DLL until Civ III calls again. There is no
completion callback, redraw request, additional timer, or renderer-owned presenter.

The camera queue also adopts a compatible in-flight or ready ambient result.
Complete view/identity proof and the existing eligible profile's quantized clock
certify reuse. It preserves the bounded future horizon and transfers a full-detail
result without a second render. The actual-worker regression holds that producer,
queues its exact bucket, verifies nonwaiting calls and one execution, then checks
pixels and rejection after visibility changes. Existing cancellation, failure,
unit takeover, configuration and reset tests remain.

Storage remains bounded: one active and one pending camera snapshot, front/ready
publications capped at 32 MiB each, and the existing 32 MiB ambient budget. During
publication the worker may temporarily own one additional capped result; reusing
an ambient frame moves its storage out of that budget before the existing immutable
publication copy. No native surfaces or pointers are handed to helpers. D3D stays
under one render-thread owner; GDI composition stays on the game thread.

The compiled bridge regression executes the actual movement, selection and queue
code, including accumulated input, clamp/wrap, reversal, barriers and failed
publication. Standalone full-detail replay compares exact pixels and ownership,
recording full request completion separately from individual simulated native
calls. Its 16 ms polling interval is a fixture parameter, **not measured Civ III
cadence**. Results and retained controls are in `retained_renderer_plan.md`.

Remaining strategic checkpoint: authorized GOG staging/install/live verification
of edge/keyboard scroll and reversal, mouse picking and overlays while pending,
unit takeover, recenter, zoom/resize, visibility/viewer/scenario changes and config
off. Actual input-to-presentation latency, native capture/blit cost and presented
cadence remain unmeasured. The user requested default activation for direct
installation/testing; this does not complete that live checkpoint.

## Earlier audit and preserved constraints

The sections below record the starting implementation and its no-ready-image gap.
The displayed-view ownership above supersedes statements that begin/poll are not
connected; source identity plumbing alone was insufficient.

## Existing path and capabilities

`patch_Map_Renderer_m71_Draw_Tiles` starts authoritative capture and runs the
original map traversal. `patch_Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`
captures the first tile pass and calls `composite_custom_renderer_frame` before
the later retained passes. The compositor calls the explicit-identity ordinary DLL render (legacy fallback for
older DLLs),
validates ownership against the current ordered callback array, and copies via
the UI-thread blit. The worker owns D3D and never touches a native canvas.
Config-off retains original rendering; custom-on map failures never replay the
native tile plane. Units retain their separate action-director/body path.

## Current caller-driven contract

The user clarified on September 12 that Civ III owns render demand: the renderer
must be ready for the next native render call, not notify Civ III when a worker
finishes. Earlier proposals for completion-driven polling/invalidation or a new
full-redraw callback are superseded. No completion notification was installed.

The CSV supplies the existing map boundaries and timer inlead. The timer remains
Civ III's existing animation scheduling path; it is not a worker-completion signal.
Completed work stays owned by the DLL until a subsequent native request can
consume a compatible result. An unchanged static camera may consume a newer
ambient publication; changed capture/camera/visibility must reject incompatible
content. An identical queued request can supply the next ordinary call; an
incompatible request still takes over synchronously. No new timer/redraw hook is a
prerequisite for this pull-based handoff. Actual native call/presentation cadence
still needs observed evidence before any native frame-rate claim.

## No-ready-image constraint

Camera and zoom changes immediately affect capture anchors, native unit/HUD
placement, native map overlays and inverse mouse picking. Their helpers read
the current injected transform; they do not read an immutable displayed-view
identity. City text also depends on the separate outstanding tile-to-screen
inlead request. An older completed bitmap cannot simply be returned while
these layers advance: image and interactions would describe different views.

The optional camera API has one active and one replaceable pending request,
plus front/ready publications. It rejects superseded tickets. At baseline it
copied pixels and ownership flags without exposing their captured occurrence
order. The candidate adds independently versioned `camera_begin_view` and
`camera_poll_view` exports. A publication now owns the exact ordered tile
records, anchors, zoom, clock and world basis, together with caller-supplied
map/viewer/visibility/scene epochs and output content/device revisions. Topology
payloads stay in preparation snapshots; the display metadata retains their
revision. Old API-17 layouts and synchronous entry points remain compatible.

The actual publication owner/worker tests cover ordering, deep copies, epochs,
failure atomicity, supersession and unit resumption. The injected caller now binds
`c3x_renderer_render_view`, using the same camera request structure for ordinary
demand with authoritative scenario/viewer/visibility identity. It does not call the
separate begin/poll entry points. Equal
ownership-array lengths are insufficient to validate a later callback ordering.
The disabled terrain-only preview remains an unaccepted full-ownership solution.

Before enabling an injected async mode, implement and test a current-camera,
coverage- and visibility-safe presentation or explicitly coordinate all retained
native layers and picking with a displayed identity. Visibility loss must reject
old content, even when a camera ticket has not changed. A subsequent native render call must consume compatible ready output without
restarting identical work. Completion alone does not cause a native render call.

The September 12 caller-driven handoff correction also validates the displayed
front against `PublishedMapFrame`'s own captured view before returning it from the
ambient legacy path. The active job's matching identity alone is insufficient:
explicit camera requests and legacy calls share the same owner. The actual worker
regression reproduces old-camera return with the former guard. The corrected
owner and small native ambient boundary pass; evidence and the one next task are
recorded only in the retained plan. That correction introduced no completion notification. The September 13 identity
connection below subsequently changes the injected bridge.

## Authoritative identity at ordinary native demand

The September 13 source implementation binds the optional `render_view` export at
existing capture/composition boundaries. Renderer unload advances scenario lifetime;
each capture certifies one native viewer. The existing world topology scan also
observes both complete native visibility words per tile, with a separate revision.
Topology supplies the scene epoch; local object/anchor changes remain covered by
exact ordered capture. Profiles without the world scan keep visibility epoch zero
and retain the local visibility checks. No complete-world appearance claim follows.

The visibility observation is bounded to eight bytes per parity tile, at most
16 MiB under existing map limits. Resize may overlap old/new observation storage;
allocation failures preserve both prior owners/count until successful commit.
This is authoritative capture bookkeeping, not a larger render cache. Executable
native capture/allocation tests, worker epoch/adoption tests, approved injected
compile and the small native explicit-identity ambient boundary pass. Evidence and
the single next task live in the retained plan.

The strategic native checkpoint remains pending staging/install/game authorization:
verify scenario/viewer/visibility transitions, passive same-view ambient demand,
exact camera/zoom with overlays and picking, and unit takeover through the real
call path. Ambient mode remains opt-in. The standalone fixture does not measure
native capture cost or presentation cadence. Before enabling a general asynchronous
camera mode, resolve its no-ready-image constraint above; source identity plumbing
alone is not that acceptance.

## Worker and memory constraints

The navigation continuation adds exact duplicate coalescing to this existing
queue. Repeated begin calls preserve the latest pending, active or completed
ticket and its ready publication when every frame field, ordered tile/topology
byte and lifecycle epoch matches. Caller pointer addresses are excluded;
padding differences can only decline reuse. A different clock, visibility,
order, world revision, camera or epoch still supersedes work. Comparison borrows
the existing immutable snapshots under the state lock and adds no snapshot
owner. Configuration, incompatible synchronous rendering, cancellation and reset
still invalidate reuse. Begin returns `PENDING`; poll consumes the preserved result.
An ordinary render can instead consume or wait for the exact request when its
caller-owned lifecycle epochs match. The legacy entry supplies zero epochs; the
new explicit-identity ordinary entry supplies the native observations. Unknown
nonzero epochs cannot be adopted by the legacy caller. This reuses the same queue, condition variable and publication
transfer, and returns only a final exact result or the queued error. It may still
block; it does not provide bounded camera-call latency or request native redraws.
The actual worker test exercises duplicates, content changes, unit takeover and
reset. The standalone camera witness now repeats begin during polling and records
submission and per-request maximum poll/duplicate-call times. These are DLL call
measurements, not game-thread scheduling or input-to-presentation measurements.

This resolves the duplicate-resubmission hazard inside the optional DLL queue.
The September 13 bridge connection supplies lifecycle epochs at ordinary demand.
It does not change when Civ III requests rendering or solve the no-ready-image
constraint above.

The first 100-request Windows queue run matched the synchronous images exactly.
Submission p95 was 0.507 ms and the per-request maximum poll-call p95 was
0.402 ms. Duplicate begin maxima had a 2.102 ms p95, narrowly missing the 2 ms
bookkeeping target. Completion copied its pixel/occurrence payload while holding
the state lock. The follow-up moves this copy and obsolete-owner reclamation
outside that lock, while `camera_active` still protects borrowed render scratch.
Only the final swap is published under the lock after rechecking the current
ticket, cancellation and pending result. A deterministic worker test stalls the
copy while identical begin, supersession and polling proceed; stale completion
cannot overwrite the new visibility epoch. The existing 32 MiB publication cap
and temporary-copy owner remain in force. With the disabled experimental preview
enabled, its ready preview can overlap the final temporary owner until the swap;
include both in the existing per-owner memory accounting.

The follow-up `navigation-completion-queue` ran 100 requests on Windows and
matched all synchronous images from the same build exactly. Submission p95 was
0.505 ms, per-request maximum poll-call p95 0.430 ms, and per-request maximum
duplicate-begin p95 0.276 ms (maximum 0.349 ms). The stressed accepted-to-complete
distribution remained 69.590 ms median / 114.569 ms p95. Its harness deliberately
starts and supersedes an obsolete environment request before each current
request and polls using `Sleep(1)`; total capture-plus-harness time includes that
extra work. This validates bounded call latency and exact completion, not an
ordinary game navigation latency or a native-presented-frame target.

At the audit baseline, `draw_unit` called `drain_camera_locked`, discarding
pending map work. The candidate now pauses camera dispatch, interrupts active
map work at its existing cancellation boundaries, preserves the latest immutable
snapshot without allocating another owner, and resumes after the UI-thread unit
copy. Newer pending input wins over the interrupted snapshot. Configuration,
incompatible synchronous map render and reset still supersede camera requests. The actual
worker test covers repeated unit takeover, rejected unit requests, eventual
latest completion and reset on MSVC. This does not yet prove the sustained-input
frame target: repeated interruptions may postpone completion, and unit drawing
still waits for already submitted GPU work. D3D ownership remains serialized.

Each `PublishedMapFrame` has a 32 MiB capacity limit including captured tile
records. Front, ready, preview-local and
replacement-local owners can overlap with renderer pixels, viewport bitmaps,
resource pixels and the GDI DIB. Active/job/pending/warm snapshot capacities and
topology copies also coexist. The per-owner cap is not a total process budget.
Deferred D3D references in the command stream are not included in the cache's
logical byte counters. Process address-space samples and tracked caches must
be reported separately; neither is a measurement of actual driver residency.

## Patch-table action

`required_user_action: []` for patch-table changes in this audit, telemetry and
explicit-identity native handoff through existing symbols. No new address or unsupported callback is assumed.
Preserve the existing separate request in `civ3_patch_dependency_ledger.md`:
`Main_Screen_Form_tile_to_screen_coords: define -> inlead`, signature
`void (__fastcall *)(Main_Screen_Form *, int, int, int, int *, int *)`, recorded
GOG `0x4E3B10`, Steam `0x4EC360`, PCGames.de `0x4E3BD0`. Its fallback leaves
native city HUD at binary native-zoom anchors. This audit does not edit the CSV
or claim that request alone enables asynchronous presentation.
