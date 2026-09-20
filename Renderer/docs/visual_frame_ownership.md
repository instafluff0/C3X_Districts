# Renderer-owned visual frames

Civ III owns gameplay, camera/visibility authority, directed actions and UI state.
The renderer consumes captured changes and owns the intervening visual frames.
It uses the existing D3D device, GPU worker and HWND presenter; no callback asks
Civ III to redraw, and no second presenter or window is introduced.

## Ownership and publication

- `UnitInstances` retains copied content, compiled catalog bindings, local
  revisions and explicit despawn retirement. Only native-selected occurrences
  can contribute; a retained instance never grants visibility. Unselected idle
  units remain frozen. Authored eligible idle/work clips use the visual clock.
  Directed actions retain their captured native cursor and anchor.
- Map samples own copied tiles, topology, projection and selected-view inputs.
  They reuse the established working extent and existing static/pass caches.
  The existing map sample buckets remain unchanged; presentation has a separate
  cadence. Each retained source owns its capture until native composition retires
  it. Preparing another map cannot freeze the displayed version or untouched
  rectangles of a partial transfer; reset/configuration retires the whole session.
- `RetainedComposition` stores ordered rectangular writes over versioned images.
  Copies reference immutable source versions, including native save/restore.
  Dynamic leaves are resident map/pose textures. Opaque writes remove covered
  history; transparent operations retain the affected underlay and exact native
  shader program. Final transfer seals only its actual rectangle, preserving the
  displayed version outside a partial transfer.
- Replay first collects direct pose revisions across the reachable graph, offering
  current immutable pose jobs to the existing CPU pool. It then evaluates dependencies,
  binds immutable source textures and
  uses cropped scratch for ordinary destination/underlay passes. Cross-position
  self-copy retains its full coordinate domain. Static results cache and release
  their recipes. Working native canvases are untouched by intervening frames.
  Unchanged source samples do not cause another display transfer.

## Clock, native integration and lifecycle

An ordinary 33 ms Win32 thread timer schedules visual opportunities on the
existing presenter's UI thread. The renderer worker samples and composes; the
same UI owner performs DXGI presentation. There is no visual-only call to
`Animator_update`, no renderer rearming of Civ III's timer, and no extra native
FLC advancement. Civ III's original gameplay timer continues unchanged.

Native unit/map captures query `c3x_renderer_visual_clock`, in QPC-frequency
units, so a later content/camera update cannot restore an older animation clock.
Existing popup/Advisor and command-button scopes supply explicit pause/resume
policy on entry and exit, including exits with no further native transfer. Final
transfer also supplies the current native visibility policy. Focus loss rebases
the visual clock. A timer cannot reenter an active renderer call. Presentation still requires the UI thread to pump messages;
this is not a separate presentation thread capable of bypassing blocked gameplay.

A complete front must exist before autonomous rendering. CPU handoff, reset and
window release stop visual delivery. Replay failure preserves the previous
completed display and withdraws the front; native compatibility demand remains
available while no front is ready. History admission can restart at a fresh
native map publication. Config-off retains the original native path.

Packed native unit scratch surfaces are valid GPU destinations even without an
optional full-color layer. Unit blending snapshots only its selected rectangle;
it must not require full-map scratch allocations. Public CPU access retains the
existing ownership barrier; a composition rejection is not permission to skip it.

Compatibility ambient redraws never run during a processed mouse press: rebuilding
native command buttons clears pressed-form ownership. No selected-unit exception
or elapsed-time guard applies. Native actions and resident visual frames continue
under their existing owners; only optional compatibility animation waits for release.

## Bounds and validation

Retained texture accounting is capped at 128 MiB, nodes at 32,768 and rectangular
patches at 8,192 per image. Separate replay scratch is capped at 128 MiB; the
live composition owner is capped at 96 MiB so 2240×1192 map/screen/popup color
pairs fit together. Generic compositor tests retain their 64 MiB ceiling. These are ceilings,
not permanent allocations. Source snapshots and CPU recipe metadata retire with
their last owning version; shared map/pose textures use COM lifetime ownership.

`test_retained_composition` compares replay against ordinary production GPU
commands, including source replacement, aliasing, native 555/565 and full color,
partial publication, opaque overwrite and reset. The connected JGL fixture
advances actual resource/unit samples without native drawing, dispatches the real
timer, verifies modal clock pause and restores exact native UI pixels. Explicit
oracle readbacks remain test-only. `visual-frame` traces report whole-call time;
read-only visual status reports frames, map/pose samples and retained ownership.

The retained plan records the final measurements and staging identity. These
fixtures do not establish coverage of every live game screen. General nonblocking
cold/outside-coverage camera updates and broader compatible unit submissions are
separate architecture responsibilities.
