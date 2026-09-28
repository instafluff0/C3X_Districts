# Custom-rendering stepped zoom

`enable_custom_rendering_zoom = true` adds stepped main-map zoom when
`enable_custom_rendering = true`. The current levels use Civ III's isometric
basis at tile widths 128, 160 and 192 pixels: 100%, 125% and 150% of native
normal size. The user limited the supported envelope to these three closest
levels on September 9, 2026; 64 and 96 are disabled in the custom zoom control.
Starting at normal, successive `Z` presses select 192, 160 and 128 pixels.
The main-map wheel now steps in either direction and clamps at these endpoints:
wheel-up moves closer, wheel-down moves outward. Small wheel deltas accumulate
until they total a Windows 120-unit notch. This GOG activation uses the audited
main-form m25 slot; config-off, loading/menu and disabled custom zoom delegate
to the original handler with unchanged arguments. Other forms are unaffected.
The projected world point at the screen center stays fixed while the level
changes. Native synchronization accepts only those three widths. A change to
Civ III's native reduced-zoom flag retains its 64-pixel anchor basis but resets
the custom view to 128 around the screen center; it does not restore a disabled
farther-out level. Configuration-off retains the original native zoom path.

Civ III remains the camera and interaction authority. The injected bridge
applies one affine scale and translation to captured map anchors, then supplies
the selected tile width to the off-screen renderer. Native events and stored mouse coordinates remain in screen pixels. The shared
`Main_Screen_Form_get_tile_coords_under_mouse` inlead applies the inverse at
picking, so down/hover/release/right-click, held-click callbacks and C3X targeting
agree. Four explicit call-site replacements preserve native map-traversal clip queries,
including incremental draws outside m71. Visible city work-area input also bypasses
the custom inverse. Per-caller inverse transforms are removed. Native map sprites, C3X tile
highlights and map text receive the same forward transform; custom-rendered
unit bodies receive an exact numeric projection scale rather than the old
normal/reduced binary. While this zoom feature is enabled, native FLC unit bodies
are never shown: if custom units are disabled or a custom body cannot render,
that body is omitted. The selected white ring shares Renderer64's sampled unit anchor and draws
before unit meshes. Native health, status and unit-HUD overlays keep their
existing ownership. Map-specific calls for city-HUD coordinates, unit
health/status, selection cursors and civilization markers transform attachment
points directly, preserving native UI size and offsets. Ordinary and army paths
are covered. Shared drawing functions and city-screen callers remain unpatched.
The general tile-to-screen function is a callable `define`; the city HUD uses a
single internal call replacement. `Unit_tick_anim` only scopes capture/canvas
ownership, with unchanged offsets and no translation/undo state. Executable tests
check call-site wiring, coordinate parity and argument preservation.

Zoom is deliberately main-map-only. The existing patched
`Main_Screen_Form_handle_key_down` boundary consumes `Z` before Civ III's
native two-level toggle. The authorized `Main_Screen_Form_process_mouse_wheel`
vtable replacement uses the same zoom implementation. The handler
retains the exact native pixel camera and uses `move_camera` to update bounds
without rounding through a tile center. It then requests a complete traversal. A plain Animator dirty
bit is insufficient because it may request only a one-tile damage redraw,
leaving the exclusive custom terrain plane without a complete visible capture.
The city screen retains its existing C3X `Z` handling. Changing Civ III's native
zoom mode with another existing control resets the custom transform to the
supported normal level around the screen center. Renderer failure retains the existing custom-map-plane policy.
Custom-on map unit failure is reported and the body is omitted; CPU unit
rasterization is not a fallback. Renderer-off units remain native.

The settler territory preview transforms only its audited native line call.
Civ III still selects the legal edges and current civilization palette color;
the GPU line pass receives that copied style. Transient `MapMessage` layout
transforms the tile attachment and translates the complete native dirty rectangle.
Its font size, overlap placement and lifetime stay native. This includes messages
from `show_map_specific_text`; shared `PCX_Image_draw_text` UI calls are unchanged.
Both hooks currently have verified GOG addresses only, recorded in the patch ledger.

Zoomed-out capture promotes only complete appearance records that can reach the
scaled viewport. The outer topology ring remains non-renderable. Intermediate
levels currently request a full map clip so native retained overlays cannot
leave stale pixels; narrower damage tracking is a later performance refinement.

The GPU tile cache still includes target size and tile width. Natural terrain
now has a separate indexed world/material mesh tier: changing zoom can reproject
its retained vertices without repeating height, coast, relief, decal or forest
construction. Reuse validates semantic, world-topology and coast dependencies;
city composition retains its ordinary compiler. Center-prioritized admission
keeps a useful working set when a wide view exceeds the cache. Natural mesh data
has a 96 MiB cap and viewport bitmaps a 32 MiB cap, reallocating the former
128 MiB bitmap allowance without increasing the combined CPU cache budget.

Vertex indexing uses a contiguous lookup table with the same exact byte equality
and first-occurrence ordering as the earlier node-based hash map. Its final hash
mix disperses regular float grids across power-of-two buckets. Chunks with at
most 65,535 vertices use lossless 16-bit triangle indices; larger chunks retain
32-bit indices. Both map and source-shadow draws honor the stored format.
Animated views
can recover an exact cached terrain bitmap; they reassemble current geometry and
resource anchors before composing poses and rebuilding any missing depth
backdrops. Posed resource pixels never enter the immutable terrain cache.

Ground passes keep their unique grid corners and triangle indices through GPU
upload, without expanding and deduplicating the triangle stream again. Mixed
object-shadow geometry retains its ordinary indexing path. Resource color/depth
backdrops now have a byte-bounded, complete-static-signature LRU across camera
views: 32 MiB normally, 160 MiB in the larger-cache benchmark tier. The current
view's blocks are not evicted by its own scan; over-budget blocks still render
normally. Only static MSAA background/depth is retained, never posed pixels.
Cache allocation failure skips admission rather than discarding a valid frame.

Continuous animated zoom is still pending. The current game adapter calls the
sandbox scene with display zoom 1 and rebuilds the captured projection for each
stepped level. The sandbox already scales its resolved terrain and moving layer
together in the final GPU pass. Reusing that approach in the game requires fog,
tactical primitives, native map HUD anchors and inverse picking to share the
same displayed transform. Scaling the final window would also scale fixed HUD
controls, so that is not an appropriate insertion point.

The native map canvas and `Units_Control` are not clean world-only layers:
`Main_GUI::FUN_00553b40` also paints fixed notification shadows into
`Units_Control`. Existing city-HUD, unit-status and marker hooks expose map
attachment points. Transient messages compute their rectangle at GOG
`0x4D7DC0` and paint at `FUN_004d7d40` (`this`, canvas, background canvas);
that painter is not currently hooked. These source findings identify the
remaining separation work; they are not an activated patch or a completed
continuous-camera implementation. Any new patch-table requirement must go
through the dependency ledger before activation.

The continuous implementation must keep these boundaries together:

- Keep a canonical captured map and ease a renderer-owned display transform
  around the viewport center. Wheel reversals begin at the current displayed
  scale. Do not generate a native map capture for every intermediate scale.
- Apply that transform to fog, terrain, units, selection, tactical primitives
  and map-HUD attachment points in the same visual sample. Fixed UI and native
  font/icon dimensions remain in screen pixels.
- Publish the last presented transform back to the bridge for inverse picking;
  the requested endpoint is not the displayed view during a transition.
- Preserve native working-image versions separately from animated display
  recipes. Native readback has exact submission semantics; evaluating an
  animated display recipe in its place would change those semantics.
- A moving HUD command must retain the background across its swept rectangle,
  and retire its previous occurrence when native drawing replaces or erases it.
  Merely changing a retained node's destination coordinates leaves stale pixels
  outside its original recorded damage region.

These are implementation constraints, not completed feature claims. The current
three-level path stays active until the complete world/UI/picking transition
passes executable overlap, erasure, reversal and live interaction checks.

For the current stepped path, `-Scenario zoom` in the
[scripted testing guide](../tools/scripted_game_test.md) covers wheel direction,
rapid reversal, partial deltas and center preservation. Historical geometry
benchmarks remain in [benchmark notes](zoom_performance.md).

Automated checks cover affine anchor invariance, inverse picking, the three-level
`Z` cycle, expanded capture, native overlay scaling and numeric unit projection.
A live checkpoint should exercise repeated `Z` steps, hover/left/right selection,
scrolling and wrapping, units and selection/status overlays, city-screen `Z`,
the native zoom control reset, and configuration-off behavior.

The regression checks include compiled injected function bodies across all three
levels, both native bases and several camera offsets; consistent event picking;
city-work-area/config-off bypass; unscaled HUD layout at a transformed attachment;
and actual CSV activation. The approved GOG input, clip-query and map-UI hooks are active;
see the patch ledger for addresses and other-build limitations.
