# Civ III Renderer Insertion Notes

## Known Existing Hooks

- `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`
  - Already patched in `injected_code.c`.
  - Provides `tile_x`, `tile_y`, `pixel_x`, and `pixel_y`.
  - Good first place to capture visible tile state and authoritative Civ III screen anchors.
- `Tile.SquareParts`
  - Existing C3X debug tile info decodes this as `sheet_index = (SquareParts >> 8) & 0xFF` and `sprite_index = SquareParts & 0xFF`.
  - Renderer mapping rules should capture both semantic terrain type and this Civ III art-variant decision.
- `Map_Renderer_m71_Draw_Tiles`
  - Already patched in `injected_code.c`.
  - Wraps the visible tile drawing pass.
  - Brackets native renderer lifetime for the main map path. The actual bitmap insertion happens inside its `m19` sequence as described below.

## Proven M5.1 Pass Boundary

The re-audit of `ref/Civ3Conquests_master.exe.c` establishes this call chain:

1. `Map_Renderer::impl_m71_Draw_Tiles` optionally clears/prepares its map image, then calls `FUN_005f4550`.
2. `FUN_005f4550` calls virtual `m20_Draw_Tiles`.
3. `Map_Renderer::impl_m20_Draw_Tiles` calls `FUN_005f4570`.
4. `FUN_005f4570` invokes virtual `m21_Draw_Tiles_by_Flags` five times, in order, with raw masks `0x9`, `0x4`, `0x1fef0`, `0x2`, and `0x100`.
5. Each `m21` traversal computes Civ III's authoritative `pixel_x`/`pixel_y` from the current camera and calls virtual `m19_Draw_Tile_by_XY_and_Flags`.

`m19` dispatches bit `0x1` to base terrain. It dispatches later bits to mountains/relief, irrigation, rivers/flood plains, forests/jungle/swamp, territory, roads/railroads, buildings, pollution, resources, cities, and the final `0x100` stage. After `m20` returns, `m71` draws the city HUD and the subsequent main-screen stage, then copies the constrained map rectangle to the caller's destination image.

The M5.1 bridge therefore uses the existing `m19` and `m71` hooks without a new address:

- `m71` opens and closes a bounded main-map capture frame only when `enable_custom_rendering` is true.
- During raw pass `0x9`, `m19` records every authoritative anchor plus valid even-parity terrain metadata.
- On the first raw `0x1fef0` call, before vanilla `m19` starts that retained-overlay pass, the bridge renders one off-screen D3D11 BGRA target, reads it back, and alpha-composites it into the existing JGL map surface.
- While the custom frame is active, the patch does not call the original `m19` for any pass. This intentionally gives the custom renderer the complete `m19` map plane during staged development; native rivers, roads, improvements, resources, cities, and other tile layers remain absent until their matching approved renderer systems arrive. Later `m71` HUD/main-screen work and the separate unit animation plane remain outside this suppression.
- If loading fails while custom rendering is enabled, the frame reports a hard diagnostic and does not enter the vanilla `m71` path. Capture, rendering, readback, validation, DC acquisition, blitting, device, and reentrant failures likewise never replay native terrain. Turning custom rendering off is the sole complete vanilla path.

This is the controlled insertion boundary: **after Civ III has prepared the map target and completed the base-anchor traversal, immediately before the `0x1fef0` retained-overlay traversal**.

## Native/Injected Ownership

- `Renderer/native/c3x_renderer.cpp` owns D3D11 device/resources, whole-frame terrain rendering, staging readback, and bounded alpha blitting.
- `Renderer/native/c3x_renderer_api.h` is the narrow versioned C ABI shared with injected C.
- `injected_code.c` owns only the configuration gate, DLL lifecycle, bounded capture, calls across that ABI, exclusive `m19` suppression, and hard-failure diagnostics.
- The native renderer creates no window, swap chain, or presenter and consumes only source-independent terrain records.

## Map Plane Versus Unit Plane

The insertion boundary above governs the retained map plane. Civ III's animated
units are not part of the `m19` tile passes: `Animator::update` later rebuilds a
visible-unit list, computes wrap-aware dynamic anchors, and calls
`Unit::tick_anim` on the separate `Units_Control` canvas with unioned dirty
rectangles. The unit metadata currently carried on a terrain record is therefore
descriptive only. M7.4 must integrate at the Animator/unit-body boundary and may
not suppress units from the terrain capture.

The audited call chain, cache-key requirements, partial-repaint rules, and
scroll/refresh failure analysis are in
[`civ3_render_loop_viability.md`](civ3_render_loop_viability.md).

## Configuration-Off Guarantee

`patch_Map_Renderer_m71_Draw_Tiles` checks `enable_custom_rendering` before calling the loader or opening capture state. When false, it invokes the original `Map_Renderer_m71_Draw_Tiles` and returns. The default and base value are false.

## In-Game Result

The three captures under `Renderer/evidence/m5_1/` are historical bridge evidence. Current I11 gives the custom surface approved base terrain, relief, water/coasts, forest/jungle bodies, dunes, and marshes with zero native fallback. Volcanoes and other unported `m19` categories are absent pending their same-numbered integrations. The separate unit plane plus later fog, borders, selection, labels, minimap, HUD, and UI stages remain outside the suppressed `m19` plane.
