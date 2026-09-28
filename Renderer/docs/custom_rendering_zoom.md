# Smooth custom-renderer zoom

`enable_custom_rendering_zoom = true`, together with
`enable_custom_rendering = true`, enables main-map wheel zoom. The endpoints
are 128, 160, 192, 224, 256, 320 and 384 pixels per tile
(1×, 1.25×, 1.5×, 1.75×, 2×, 2.5× and 3×). Wheel-up moves
closer; wheel-down moves outward. Small Windows wheel deltas accumulate to a
120-unit notch. Z cycles outward and wraps. The viewport center stays fixed.
Configuration-off, loading/menu and disabled custom zoom delegate to Civ III's
original wheel function with unchanged arguments.

## One canonical world, one displayed transform

The game captures map, unit, label and tactical coordinates on a stable
128-pixel basis. Civ III's reduced-zoom mode still supplies 64-pixel anchors;
the existing center-preserving capture transform converts those to 128.
Wheel input publishes only a copied Q16 target through the existing async image
queue. It does not recapture terrain, invalidate world assets, call the native
camera setter or wait for the renderer.

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
attachment at the same sampled zoom. Fixed panels retain their screen position
and original size. All retained native UI backgrounds refer to one current
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
World-map custom zoom resumes after Civ III clears its spotlight on exit.

## Picking and lifecycle

Only a successful DXGI `Present` publishes the Q16 display scale to the helper's
shared header. Both mouse coordinates use one atomic sample of that scale, then
undo the existing native/canonical transform. Reading it submits no RPC and
waits for no frame. Native map-clip queries retain their separate untransformed
call sites; native city-screen input retains its native camera.

The helper wire is version 11. The successful-presentation counter remains at
byte 228; the presented zoom scale is at byte 232. The bridge, helper and x64
DLL must be built and staged together. Scene teardown releases the view,
notification snapshots and retained images; injected teardown resets wheel
remainder and canonical/target state before another map is loaded.

The current 3× extension is a standalone candidate. The prior live evidence
below covers the earlier 1.5× limit; it does not qualify the extended range in
Civ III. Bridge target admission, helper telemetry and GPU image transforms use
the same limits as scene projection. City-view restrictions are unchanged.

## Verification

```sh
python3 -m unittest Renderer.native.test_custom_zoom \
  Renderer.native.test_zoom_transition Renderer.native.test_zoom_integration \
  Renderer.native.test_zoom_gpu Renderer.native.test_retained_composition \
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
