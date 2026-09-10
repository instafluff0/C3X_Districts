# Native asynchronous presentation audit

Current source audit, September 9, 2026. No hooks were installed, patch-table
addresses changed, or game launched for this audit. This is an implementation
constraint record, not a live presentation pass.

## Existing path and capabilities

`patch_Map_Renderer_m71_Draw_Tiles` starts authoritative capture and runs the
original map traversal. `patch_Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`
captures the first tile pass and calls `composite_custom_renderer_frame` before
the later retained passes. The compositor calls the synchronous DLL render,
validates ownership against the current ordered callback array, and copies via
the UI-thread blit. The worker owns D3D and never touches a native canvas.
Config-off retains original rendering; custom-on map failures never replay the
native tile plane. Units retain their separate action-director/body path.

The CSV supplies both map vtable replacements and the existing timer inlead.
`Animator_update` is `define` (callable), not an entry patch. The current timer
sets the existing animator dirty byte before calling the original approximately
66 ms callback. It is gated by visible animation, focus/map/modal/input state
and an outstanding-redraw flag. No completion notification is currently wired
to that path. A static-camera completion therefore has no guaranteed redraw.

The custom zoom handler already calls `Main_Screen_Form_bring_tile_into_view`
with refreshed tile bounds. Its comments and zoom contract explain why the
animator dirty byte alone is insufficient: it can produce a one-tile damage
traversal. A completion handler must request full enough retained-map work on
the UI thread without recursing into the capture it is completing. Calling this
camera operation from a worker or treating a window repaint as a proven full map
capture would violate the current boundary. The audited timer can support a
coalesced low-frequency poll/invalidation experiment; it cannot establish 30 Hz.
A higher-frequency UI callback and its full-redraw semantics still need proof.

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
failure atomicity, supersession and unit resumption. The injected caller does
not yet bind this extension or supply authoritative lifecycle epochs. Equal
ownership-array lengths are insufficient to validate a later callback ordering.
The disabled terrain-only preview remains an unaccepted full-ownership solution.

Before enabling an injected async mode, implement and test a current-camera,
coverage- and visibility-safe presentation or explicitly coordinate all retained
native layers and picking with a displayed identity. Visibility loss must reject
old content, even when a camera ticket has not changed. A completion-only redraw
must consume compatible output without repeatedly submitting identical work.

## Worker and memory constraints

The navigation continuation adds exact duplicate coalescing to this existing
queue. Repeated begin calls preserve the latest pending, active or completed
ticket and its ready publication when every frame field, ordered tile/topology
byte and lifecycle epoch matches. Caller pointer addresses are excluded;
padding differences can only decline reuse. A different clock, visibility,
order, world revision, camera or epoch still supersedes work. Comparison borrows
the existing immutable snapshots under the state lock and adds no snapshot
owner. Configuration, synchronous rendering, cancellation and reset still
invalidate reuse. Begin returns `PENDING`; poll consumes the preserved result.
The actual worker test exercises duplicates, content changes, unit takeover and
reset. The standalone camera witness now repeats begin during polling and records
submission and per-request maximum poll/duplicate-call times. These are DLL call
measurements, not game-thread scheduling or input-to-presentation measurements.

This resolves the duplicate-resubmission hazard inside the optional DLL queue.
It does not bind the extension in injected code, provide lifecycle epochs, add
a native completion callback, or solve the no-ready-image constraint above.

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
synchronous map render and reset still supersede camera requests. The actual
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

`required_user_action: []` for the source audit, telemetry and renderer-only
prepared-view work. No new address or unsupported callback is assumed.
Preserve the existing separate request in `civ3_patch_dependency_ledger.md`:
`Main_Screen_Form_tile_to_screen_coords: define -> inlead`, signature
`void (__fastcall *)(Main_Screen_Form *, int, int, int, int *, int *)`, recorded
GOG `0x4E3B10`, Steam `0x4EC360`, PCGames.de `0x4E3BD0`. Its fallback leaves
native city HUD at binary native-zoom anchors. This audit does not edit the CSV
or claim that request alone enables asynchronous presentation.
