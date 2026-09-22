# Renderer-owned visual frames

Civ III owns gameplay, camera/visibility authority, directed actions and UI state.
The renderer consumes captured changes and owns the intervening visual frames.
It uses the existing D3D device, GPU worker and existing HWND; no callback asks
Civ III to redraw, and no second presenter or window is introduced.

## Ownership and publication

- `UnitInstances` retains copied content, compiled catalog bindings, local
  revisions and explicit despawn retirement. Only native-selected occurrences
  can contribute; a retained instance never grants visibility. Unselected idle
  units remain frozen. Authored eligible idle/work clips use the visual clock.
  Directed actions retain their captured native cursor and anchor. A new native
  action may supersede the displayed selection before the next screen transfer;
  that already-published pose stays frozen until replaced, without invalidating
  the water/resource composition around it.
- Map samples own copied tiles, topology, projection and selected-view inputs.
  They reuse the established working extent and existing static/pass caches.
  The existing map sample buckets remain unchanged; presentation has a separate
  cadence. Each retained source owns its capture until native composition retires
  it. Preparing another map cannot freeze the displayed version or untouched
  rectangles of a partial transfer; reset/configuration retires the whole session.
- `RetainedComposition` stores ordered rectangular writes over versioned images.
  Copies reference immutable source versions, including native save/restore.
  Equal-coordinate, same-format copies share those versioned patches directly;
  they do not allocate or evaluate another full-screen replay texture. Shifted,
  converted and procedural copies retain their ordered execution path.
  Dynamic leaves are resident map/pose textures. Opaque writes remove covered
  history; transparent operations retain the affected underlay and exact native
  shader program. Final transfer seals only its actual rectangle, preserving the
  displayed version outside a partial transfer.
- GPU camera assembly uses the existing one-active/one-pending camera queue.
  Pending replacement owns copied inputs and a separate completed texture; it
  cannot replace the adopted map until a successful poll or synchronous demand
  adopts it. Native image work pauses/resumes assembly and preserves that
  request. Cancellation retires prospective output without discarding durable
  world changes. The existing synchronous native barrier remains until atomic
  view publication and coherent overlays/picking are implemented in M3.4–3.6.
- Replay first collects direct pose revisions across the reachable graph, offering
  current immutable pose jobs to the existing CPU pool. It then evaluates dependencies,
  binds immutable source textures and
  uses cropped scratch for ordinary destination/underlay passes. Cross-position
  self-copy retains its full coordinate domain. Static results cache and release
  their recipes. Working native canvases are untouched by intervening frames.
  Unchanged source samples do not cause another display transfer.

A first valid native line stroke is destination demand, like the existing
HUD blend/copy path. It admits an eligible canvas before native line setup can
open a public DC; merely querying line capability does not allocate. Actual
escaped CPU/DC aliases still require the original native fallback.

Static color/depth under animated damage uses small original-format MSAA backup
tiles. Only touched tiles are resident; camera replacement retires unused tiles.
Every color/depth sample is preserved, with no change to resolve or quality.

## Clock, native integration and lifecycle

An owned cadence thread offers one visual frame at a time, targeting 33 ms with
at least 10 ms between completion and the next opportunity. There is no catch-up
queue. It tries the existing renderer transaction gate, yields to camera/native
work, and submits sampling/composition to the existing D3D worker. It then
presents the completed composition swap chain independently of the game message
pump. No new window, D3D context, native draw or game-state read is introduced.
Civ III's original gameplay timer and directed action advancement are unchanged.

The composition visual replaces the old HWND-bound swap chain on Civ III's own
window. A composition swap chain has no HWND, avoiding the UI-message dependency
of HWND presentation. Creation, first attachment and native ownership changes
stay on the UI caller; subsequent ambient Present uses DO_NOT_WAIT. Backpressure
retains a pending presentation for a later opportunity. Presentation failure
retires readiness and re-enables native compatibility recovery. Detaching the visual and
waiting for that detach restores native GDI; it is below native child windows.
This path requires Windows 8 or later (validated on Windows 11). It is distinct
from putting a flip-model HWND swap chain on Civ III's GDI window.

Native screen transfers commit their changed rectangle and sample the same
retained scene at the current visual clock before presentation. They cannot
restore the immutable map's original water/resource time between autonomous
frames. This also supplies ambient samples while repeated native transactions
occupy the caller gate. Explicit clock suspension retains the static native
transfer path; holding a command or advancing native actions is not suspension.
Camera preparation and adoption retain the existing scheduling guard. Visible eligible
unit loops advance by elapsed time modulo clip duration even after a slow frame;
unselected idle units and explored-but-not-visible content remain frozen.

Native unit/map captures query `c3x_renderer_visual_clock`, in QPC-frequency
units, so a later content/camera update cannot restore an older animation clock.
Popup/Advisor and command-button scopes guard optional native redraws, without
pausing the renderer clock. A visible, non-minimized window continues ambient
delivery even when another window has focus. An interturn UI stall does not
pause an eligible front: water, resources and authorized idle/work poses use copied
scene/visibility records while native actions retain their last observed cursor.
Camera, viewer and lifecycle changes still validate/adopt coherently. Native
handoff disables delivery under the same gate; reset joins the cadence thread
before acquiring that gate and shutting down the D3D worker.

A complete front must exist before autonomous rendering. CPU handoff, reset and
window release stop visual delivery. Replay failure preserves the previous
completed display and discards the failed recipe and its allocations; native compatibility demand remains
available while no front is ready. History admission can restart at a fresh
native map publication. This is recipe recovery on a healthy device, **not**
recovery of authoritative native GPU pixels after device removal. A removed
device cannot satisfy the CPU ownership barrier; never expose stale native pixels
as a fallback. Config-off retains the original native path on a healthy device.

Readiness includes the map's ambient dependency, not merely completed screen
pixels or an animated unit. If an animated map becomes a full CPU snapshot,
the DLL reports recovery demand until native map writes restore a reachable
animated map source. Copied and transparent composition nodes propagate that
map dependency; a genuinely static map does not require an animation callback.

Packed native unit scratch surfaces are valid GPU destinations even without an
optional full-color layer. Unit blending snapshots only its selected rectangle;
it must not require full-map scratch allocations. Public CPU access retains the
existing ownership barrier; a composition rejection is not permission to skip it.

Compatibility ambient redraws never run during a processed mouse press: rebuilding
native command buttons clears pressed-form ownership. No selected-unit exception
or elapsed-time guard applies. Native actions and resident visual frames continue
under their existing owners; only optional compatibility animation waits for release.

## Bounds and validation

Retained texture accounting is capped at 256 MiB, nodes at 32,768 and rectangular
patches at 8,192 per image. Separate replay scratch is capped at 128 MiB; the
live composition owner is capped at 256 MiB. Eight packed/full-color native
canvas pairs plus old/new immutable maps need about 194 MiB at 2240×1260; the
live capture exhausted the previous 128 MiB cap. The remaining allowance covers
small UI sources. Generic compositor tests retain their 64 MiB ceiling. These are ceilings,
not permanent allocations. Replay recycles up to 32 MiB / 32 working textures
inside its existing 128 MiB cap, evicting idle scratch before admission failure.
Only private scratch is recyclable; published snapshots and borrowed sources
retain their original lifetimes. Reset/discard clears the pool. A complete
disjoint picture may copy its fragmented base once and overlay later results;
sparse pictures, zero patches and aliased unions keep exact rectangle copies.
Replay checks replacement capacity before allocating
a new result and releases temporary image handles even when final sampling or
display throws. Visible scene storage has no unused off-screen guard padding.
Once-per-second `process-memory` records report whole-process available virtual
memory without a VirtualQuery walk; failure records also report the device HRESULT. Source snapshots and CPU recipe metadata retire with
their last owning version; shared map/pose textures use COM lifetime ownership.

`test_retained_composition` compares replay against ordinary production GPU
commands, including source replacement, aliasing, native 555/565 and full color,
partial publication, opaque overwrite and reset. The connected JGL fixture
advances actual resource/unit samples without native drawing, blocks the window thread for two seconds while checking actual desktop motion,
verifies explicit clock suspension and restores exact native UI pixels. Explicit
oracle readbacks remain test-only. `C3X_RENDERER_MANUAL_VISUAL=1` is a replay-only
control for direct-call timing; the blocked-UI test clears it, and production
leaves it unset. `visual-frame` traces report whole-call time;
read-only visual status reports frames, map/pose samples and retained ownership.

The retained plan records the final measurements and staging identity. These
fixtures do not establish coverage of every live game screen. General nonblocking
cold/outside-coverage camera updates and broader compatible unit submissions are
separate architecture responsibilities.
