# Custom-rendering stepped zoom

`enable_custom_rendering_zoom = true` adds stepped main-map zoom when
`enable_custom_rendering = true`. The current levels use Civ III's isometric
basis at tile widths 64, 80, 96, 112 and 128 pixels. Each `Z` press steps from
128 toward 64, then wraps from 64 to 128. The projected world point at the
screen center stays fixed while the level changes.

Civ III remains the camera and interaction authority. The injected bridge
applies one affine scale and translation to captured map anchors, then supplies
the selected tile width to the off-screen renderer. Mouse coordinates are
inverse-transformed before native tile picking. Native map sprites, C3X tile
highlights and map text receive the same forward transform; custom-rendered
unit bodies receive an exact numeric projection scale rather than the old
normal/reduced binary. While this zoom feature is enabled, native FLC unit bodies
are never shown: if custom units are disabled or a custom body cannot render,
that body is omitted. Native selection, health, status and unit-HUD overlays keep
their existing ownership. The unit animation hook shifts its native screen
offset so the body and status layers share the transformed unit center. The
native city-HUD pass enables a narrow context in which the existing
`Main_Screen_Form_tile_to_screen_coords` hook transforms city-label anchors.

Zoom is deliberately main-map-only. It uses the existing patched
`Main_Screen_Form_handle_key_down` boundary and consumes `Z` before Civ III's
native two-level toggle. No mouse-wheel vtable entry is required. The handler
uses Civ III's native `bring_tile_into_view` operation to preserve the map center
and force the same complete tile traversal as native `Z`. A plain Animator dirty
bit is insufficient because it may request only a one-tile damage redraw,
leaving the exclusive custom terrain plane without a complete visible capture.
The city screen retains its existing C3X `Z` handling. Changing Civ III's native
zoom mode with another existing control resets the custom transform to that
native level. Renderer failure retains the existing custom-map-plane policy.
Outside this zoom mode, unit-body failure retains the existing native fallback;
while zoom is enabled, the failed body is omitted as described above.

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

This does not yet provide continuous animated zoom: GPU uploads and non-natural
geometry still depend on the target projection. The next architectural step is
retaining projection-independent GPU meshes, then presenting a resampled retained
bitmap immediately while a high-quality target raster completes. Performance and
pixel parity can be reproduced with `Renderer/native/BENCHMARK_ZOOM.bat` and
`Renderer/native/compare_zoom_benchmark.py`; see [benchmark notes](zoom_performance.md).

Automated checks cover affine anchor invariance, inverse picking, the five-level
`Z` cycle, expanded capture, native overlay scaling and numeric unit projection.
A live checkpoint should exercise repeated `Z` steps, hover/left/right selection,
scrolling and wrapping, units and selection/status overlays, city-screen `Z`,
the native zoom control reset, and configuration-off behavior.
