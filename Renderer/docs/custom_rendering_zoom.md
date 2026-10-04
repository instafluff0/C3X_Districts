# Smooth custom-renderer zoom

`enable_custom_rendering_zoom = true`, together with
`enable_custom_rendering = true`, enables main-map wheel zoom. The endpoints
are 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 and 384 pixels per tile:
**0.5×, 0.625×, 0.75×, 0.875×, 1×, 1.25×, 1.5×, 1.75×, 2×, 2.5×, 3×**.
There are eleven levels; a new main-map view starts at vanilla default 1×. Wheel-up moves
closer; wheel-down moves outward. Small Windows wheel deltas accumulate to a
120-unit notch. Z cycles outward and wraps. The viewport center stays fixed.
Configuration-off, loading/menu and disabled custom zoom delegate to Civ III's
original wheel function with unchanged arguments.

## One canonical world, one displayed transform

The game captures map, unit, label and tactical coordinates on a stable
128-pixel basis. Civ III's reduced-zoom mode still supplies 64-pixel anchors;
the existing center-preserving capture transform converts those to 128.
Wheel input publishes only a copied Q16 target through the existing async image
queue without waiting for the renderer. Changes entering, leaving or staying
below 1× also refresh visible capture and unit representatives. That capture
uses a separate 0.5× traversal envelope; native `TileX/Y_Min` remain unchanged
because Civ III uses them as projection origins. Resident world assets remain
valid. Camera limits follow the successfully displayed scale.

Renderer64 owns an elapsed-time, critically damped transition. Retargeting
preserves velocity, so reversing the wheel decelerates naturally. A step reaches
99% of its destination in about 166 ms. Animation, map and UI publication keep
using the existing renderer worker and presentation cadence.

The injected JGL transfer boundary identifies the actual main-map and
`Units_Control` canvases by their native owners. The worker selects their
complete retained source versions. The displayed scale is applied while the
shared scene rasterizes geometry, so closer views acquire new geometric detail.
Map overlays compose over that projected scene before fixed-size native HUD.
Immutable native images keep their GPU image transform. It rebuilds this world from canonical sources even when
Civ III requests a dirty subrectangle; an older zoomed screen is never used as
the next world's input. Fixed GUI forms compose afterward in screen coordinates.
Only Renderer64 graphics grow with the map. Native city labels, unit status
icons and map messages retain their original pixel dimensions. Their scoped
drawing commands replay after world scaling, translated by their canonical
attachment at the same sampled zoom. At outward zoom, off-canvas native HUD ink is laid out inside its existing
surface, then replayed at its original world attachment. This avoids native
surface clipping without enlarging canvases or scaling text. Captured tile
occurrences place outer city labels; the outer traversal supplies unit status
that the native Animator's narrower traversal does not visit. Fixed panels
retain their screen position and original size. All retained native UI backgrounds refer to one current
world selection, including its map HUD. A new camera replaces that selection;
old city-label placements cannot survive in earlier native screen fragments.
Native UI pixels and partial screen publication retain Civ III's normal dirty
rectangles, so unchanged panels remain visible. Canvas replacement copies and
clears retire covered HUD captures. HUD capture admits the actual Map_Renderer
canvas as well as Main_Screen_Form and Units_Control canvases.

A busy renderer transaction rejects an optional display offer immediately. The
existing cadence retries that busy case after 2 ms instead of discarding the
whole 16.7 ms frame opportunity. Completed and unchanged static frames keep
the normal period; no old timestamp or frame is queued.

Native working images remain canonical. The display recipe carries both its
full-color image and the matching packed native words, so subsequent native
keyed UI reads the correct displayed underlay. No CPU map readback or software
map fallback is used.

Civ III writes notification text into the GUI canvas but writes its shadows
into `Units_Control`. `patch_Main_GUI_draw_notifications` gives those shadow
writes an explicit lexical scope. The worker preserves their lookup tables,
excludes their rectangles from the world image, and reapplies the shadows over
the displayed world before GUI text. See the exact GOG hook and other-build
limitations in the [patch ledger](civ3_patch_dependency_ledger.md).

## City screen

Civ III's `Map_Renderer.spotlight_on_city` excludes the modal city screen from
custom zoom, HUD capture and world-view transformation, including its first
centering draw before the form is visible. Native Z keeps the two 64/128-pixel
tile sizes and calls the existing city-center routine. Manual panning is ignored;
exact centering for opening, switching cities and native Z remains active.
The custom path scales the city-center row offset with the native tile size,
keeping its pixel attachment unchanged. Both offsets are even, avoiding native
tile-parity correction of the horizontal camera. Config-off keeps the existing
C3X city-centering behavior.
The existing C3X `auto_zoom_city_screen_for_large_work_areas` setting still
opens the city at native 0.5× when its effective configured work radius is at
least four. Work-area limits retain their existing effect. City Z uses only
native 0.5× and 1×; it never changes the eleven-level world target.
World-map custom zoom resumes after Civ III clears its spotlight on exit.

## Minimap and edge scrolling

The minimap box uses the visible viewport at the last successfully presented
zoom, rather than the larger tile traversal rectangle. Zoom changes request a
GUI refresh even when the map is idle. Native minimap clicking and seam splitting
remain in place. When the viewport covers the entire map, no rectangle is drawn.
When just one viewport axis covers the world, the outline spans that minimap axis with a three-pixel inward offset so the decorative
frame cannot hide it. Native minimap state is restored immediately afterward.
Non-wrapping camera limits use the same inverse projection,
so northern/southern terrain remains reachable at close zoom. Zooming outward
re-clamps a camera that used those expanded limits.

Late callbacks use a capped 50 ms movement step instead of disabling scrolling
when the game thread takes more than 200 ms. The first callback establishes the
clock; modal, interturn and combat blocks remain unchanged.

Custom rendering replaces the native integer edge jumps with an independent
16 ms UI-thread timer. The outer 32 pixels form a quadratic speed ramp: motion
starts slowly on entering the band and speeds up toward the edge. The existing
three native scroll-speed settings select maximum speeds of 450, 900, or 1800
display pixels/second. Zoom compensates camera displacement to preserve screen
speed. Fractional pixels carry between ticks; long stalls, focus loss, dialogs,
city screens and interturn pauses cannot accumulate a later jump. This is a
timer target, not a claim of 60 rendered frames/second; actual cadence still
depends on native composition and the renderer. Timer steps let an outstanding
camera frame finish instead of repeatedly replacing it; explicit camera jumps
and changes to authoritative scene identity still supersede it.

The camera continues to supply authoritative anchors and HUD placement. Native
animation and turn timer intervals are unchanged. Config-off delegates to vanilla.
The extra timer is installed only when the audited native scroll hook is available.

An entirely unrevealed viewport is a valid empty scene. Visibility reuse accepts
zero contributors, and shadow setup uses a finite neutral receiver extent so
zooming or scrolling through black map space can continue and return to terrain.
See `Renderer/tools/scripted_game_test.md` for the `navigation` capture scenario.

## Picking and lifecycle

Only a successful DXGI `Present` publishes the Q16 display scale to the helper's
shared header. Both mouse coordinates use one atomic sample of that scale, then
undo the existing native/canonical transform. Reading it submits no RPC and
waits for no frame. Native map-clip queries retain their separate untransformed
call sites; native city-screen input retains its native camera.

The helper wire is version 14. The successful-presentation counter remains at
byte 228; the presented zoom scale is at byte 232. The bridge, helper and x64
DLL must be built and staged together. Scene teardown releases the view,
notification snapshots and retained images; injected teardown resets wheel
remainder and canonical/target state before another map is loaded.

The 0.5×–3× range shares the same admission limits across all renderer paths.
Earlier evidence below covers the prior 1×–3× and 1×–1.5× ranges. Bridge target
admission, helper telemetry and GPU image transforms use the same limits as
scene projection. City-view restrictions are unchanged. Dynamic border culling uses the inverse
displayed viewport, so outward zoom includes complete coastal border segments.

Shadow placement admission keeps the existing 32 MiB joint limit. Optional
shadow proof/page and merged-terrain caches are released if they block required
body or shadow placements, followed by one retry. Body and shadow generations
carry only their own pass ranges. A failed fresh render keeps the last completed
view but no longer permanently rejects later camera requests.

## Verification

The 2026-10-04 `085348` disposable navigation capture completed all 13 mouse
events and four accepted/adopted camera routes with no renderer failure or save
change. At 2240×1260, the 3× visible extent was 746×420 native pixels; the camera
reached Y = −420 and 1112, crossed the horizontal seam, and returned to revealed
terrain. Window samples show the minimap shrinking/expanding and terrain returning.
The approved injection smoke test and 69 host tests passed; staged trio hashes
and game/helper cleanup were checked.

In the same north-scroll segment, median camera adoption intervals were
128.6 ms before scroll-request coalescing and 90.7–99.7 ms afterward (10 Hz
window capture in the VM). These are camera update intervals, not scanout FPS.
Native GUI composition still limits fluidity; the 16 ms timer does not establish
60 FPS scrolling.

```sh
python3 -m unittest Renderer.native.test_custom_zoom \
  Renderer.native.test_zoom_transition Renderer.native.test_zoom_integration \
  Renderer.native.test_zoom_gpu Renderer.native.test_zoom_out \
  Renderer.native.test_camera_navigation Renderer.native.test_retained_composition \
  Renderer.native.test_async_publication Renderer.native.test_scripted_game_input
```

The clock tests cover frame-rate independence, rapid reversal, endpoints,
repeated targets, reset and presented-only picking. Compiled injected wrappers
cover wheel delegation, unchanged canonical capture, one display sample per
mouse point and balanced notification scopes. GPU checks compare intermediate
pixels, native 555/565 keying, fixed panels, marker alignment, source isolation,
partial native copies and publications, exact paired quantization, repeated world
boundaries and invariant native HUD ink through translated intermediate/reversed
views. Existing compositor
oracles remain in the suite.

Use the bounded workflow in [scripted game testing](../tools/scripted_game_test.md).
The zoom scenario sends ten wheel inputs, including a rapid reversal and three
40-unit deltas. With `-MeasureCadence`, it samples the presentation counter and
Q16 zoom value every 20 ms during that scenario, without a renderer command.
Use actual window capture timestamps to inspect intermediate views; counters
alone do not prove visible correctness. A one-Hz capture is the performance
control; ten-Hz capture is a bounded visual diagnostic. Follow zoom with the
scroll and interaction scenarios to check terrain, units, transient map text
and fixed UI after changing view.

Live zoom capture `20260928-083851` passes all ten wheel inputs, intermediate
scales, reversal and small-delta accumulation. Window review confirms aligned
terrain, units and selection. HUD capture `20260928-080539` verifies fixed-size
labels with no stale copies after scrolling; city capture `20260928-082829`
verifies an unchanged city anchor at both native zoom levels and ignored wheel/
edge-scroll input. The subsequent graphics-quality work replaces image enlargement
with shared geometry projection; see [quality and validation](render_quality.md).
The [current status](retained_renderer_plan.md) records the installed candidate,
measured live FPS and unresolved cold city-view delay. Earlier stepped-camera
measurements remain in [zoom performance](zoom_performance.md) for comparison.

The city zoom diagnostic records the nearest matching tile from the native
capture. Civ III's standalone tile-to-screen helper selects the wrong half-world
copy when the viewport exceeds half of a small map; that helper alone cannot
verify the city's displayed anchor. The settled city images still use native
64/128-pixel tiles. Cold 0.5× city geometry preparation can briefly leave the map
blank and remains a performance limitation.

Before assembling a different camera selection, the completed fresh passes release
their borrowed mesh memberships. Published images remain available while the new
selection is prepared. This prevents the previous wide view from keeping evicted
geometry charged against the next view's cache budget. Independent consumers still
retain their own leases; `test_scene_membership.py` checks both lifetimes.

Busy-world GOG capture `20261004-112852` completed all 15 mouse events, the Z
toggle, 17 edge-scroll updates and the return to 1×. The full trace contains no
camera, fresh-pass or tile-cache admission failure and no dropped records.
Window review verifies continuous coastal borders, attached unit/city HUD and
the changing minimap rectangle. The original save remained unchanged. This
profiled visual diagnostic is not an FPS benchmark; dense-map performance
remains a separate limitation, and reduced geometry LOD has not been enabled.

City capture `20261004-113215` confirms native 64/128 tile sizes and correctly
centered settled images. Its old anchor-only assertion failed because it used
the native half-world helper; the diagnostic now reads the captured occurrence.
The corrected logging branch passes the executable host test and approved injected
compile. Its final live repeat and the small-map repeat are pending: a disposable
diagnostic crash left a terminated Windows process holding the game's instance
state, so the newly installed game exits before loading. The original saves are
unchanged. This does not qualify those pending repeats as passes.

## Native centering and combat input

Vanilla centers a newly selected unit when it falls outside its comfortable
on-screen area; it does not center every already-visible selection. Both manual
selection and automatic cycling retain that native policy. The native visibility
query now uses the presented zoom's visible bounds and proportional screen
margins, restoring the full capture bounds on return. Native parity, wrapping,
selection eligibility and exact centering calls are retained. A changed selection
also discards an unfinished manual pan, including when the new unit needs no
camera jump.

The scroll timer respects the full existing combat scope, disabled native GUI
input and the directed-animation flag. The Animator discards pending manual
navigation during combat instead of adopting its old destination. Vanilla can
still center the fight. Its animation loop skips normal window dispatch, so the
existing Animator hook polls only wheel/Z zoom input during combat and updates
the minimap at the presented scale. Other keys, clicks and timers stay with
native processing. Zoom publishes renderer targets and requests outer capture when crossing below 1×. Fractional scroll
motion is cleared while blocked, and display-pixel speed is compensated using
the presented zoom during simultaneous wheel transitions and edge scrolling.

Executable checks cover native 1x visibility parity, cropped-off units at 3x,
wrapped visible copies, centering superseding queued pans, selection cancellation,
combat with no queued clip, disabled input, and continuous zoom reversals during
scrolling. The disposable `camera` scenario supplies the live combat/input witness.

GOG capture `20261004-093846` completed 17 mouse events and a native fight.
The camera centered at `(2400, 810)` and stayed there through four wheel changes
and Z; no edge-scroll event occurred inside combat. Presented telemetry and
window samples confirm the 2x/3x changes and the resized minimap. Combined
zoom/scroll resumed afterward, including wrapped unrevealed terrain. The run
continued presenting through its 110-second limit with no renderer failure,
939 window samples and an unchanged source save. The approved injected compile
and 72 focused host tests pass. This verifies input ordering and visible zoom,
not a guarantee of 60 FPS scrolling.

The combat-odds HUD retains its saved background as a small native image copied
through the existing GPU image hooks. It no longer requests CPU pixels from the
GPU-owned map after combat; custom-off keeps the original pixel buffer.
