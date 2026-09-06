# Civ III Render-Loop Viability Audit

## Verdict

The off-screen renderer strategy is viable, with one required architectural
clarification: Civ III has a retained map plane and a separately refreshed unit
plane. Terrain and tile-bound replacements belong at the proven
`Map_Renderer::m71`/`m19` boundary. Animated unit bodies cannot be treated as
another field on a captured terrain tile; they must integrate with Civ III's
`Animator::update`/`Unit::tick_anim` path and its dirty rectangles.

The current M6 terrain bridge is correctly placed. It renders no independent
window or swap chain and uses Civ III's own map target and authoritative
anchors. During the reopened live diagnostic it deliberately owns the complete
`m19` map plane, so retained tile layers do not run; later `m71` work and the
separate unit plane remain Civ III-owned. The M6.7 implementation has
a correctness-first, single-entry full-fragment cache: reuse requires an exact
content fingerprint, while dirty flags remain hints only. Richer retained or
fragment caching may be added only under the invalidation rules below.

## Actual Main-Map Call Chain

The decompiled executable establishes this normal map update:

1. `Animator::refresh` calls `Animator::update` when the camera or an existing
   animator dirty flag requires work.
2. On a camera/full-map refresh, `Animator::update` refreshes the units canvas,
   records the camera, calls `Map_Renderer::m71_Draw_Tiles`, then performs the
   remaining map/UI refresh work.
3. `m71` prepares its retained JGL map image, calls `m20`, draws later city-HUD
   and main-screen work, and copies its constrained rectangle to the caller.
4. `m20` reaches `m21_Draw_Tiles_by_Flags` five times, in this order:
   `0x9`, `0x4`, `0x1fef0`, `0x2`, `0x100`.
5. Each `m21` traversal derives screen anchors from `camera_x`, `camera_y`, the
   visible tile bounds, zoom-dependent 128x64 or 64x32 tile bases, and the
   camera remainder. It passes canonical wrapped tile coordinates but preserves
   the chosen on-screen occurrence in `pixel_x`/`pixel_y`.
6. `m19_Draw_Tile_by_XY_and_Flags` dispatches base terrain (`0x1`), relief,
   irrigation, rivers, vegetation, territory, roads/rails, buildings,
   pollution, resources, cities, and final retained work in that order.

The custom bridge captures the `0x9` traversal and composites immediately before
the first `0x1fef0` item. While its frame is active it omits the original `m19`
call for every pass, intentionally hiding all native tile-plane categories until
their approved custom systems arrive. This preserves Civ III's target, clip,
camera math, and later presentation while making a stale or all-native renderer
path immediately visible.

## Actual Unit Update And Draw Chain

Units are not drawn by the terrain traversal. After any necessary map refresh,
`Animator::update` performs a separate dynamic update:

1. It rebuilds a visible-unit list in Civ III's own stack/display order,
   canonicalizing wrapped coordinates and applying visibility and city/stack
   selection rules.
2. It advances FLC animations and calculates the correct wrapped camera origin
   through the routine decompiled as `Animator::FUN_004f0b90`.
3. For each visible unit it derives a body/HUD dirty rectangle with
   `Unit::FUN_005cbc30`, calls `Unit::tick_anim` on
   `main_screen_form.Units_Control.Data.Canvas`, and unions the affected region.
4. `Unit::tick_anim` draws the current-unit cursor/underlay, the FLC body, and
   health/activity/stack status work in a single routine. The body itself uses
   direct sprite blits rather than `Sprite::draw_on_map`.
5. Civ III presents the unioned unit/effect region without rebuilding the
   static map when the camera and map remain unchanged.

Consequences for M7.4:

- Unit records captured while traversing terrain are descriptive preview/export
  data only. They lack the visible-list order, authoritative interpolated pixel
  position, clip phase, event identity, and complete stack needed for rendering.
- Custom unit capture and composition must occur in the Animator unit plane.
  It must use the same wrapped origin and dirty rectangle Civ III chose for that
  update.
- The eventual suppression hook must omit only the FLC body call. The native
  cursor, health/activity/status work, dirty-region accounting, gameplay waits,
  and unit lifecycle remain intact.
- M7.4 must first prove an exact body-only boundary. `Animator_update` is
  currently callable but not entry-patchable; `Unit::tick_anim` and its direct
  body blits remain audit candidates. No new CSV request is justified until that
  spike selects the smallest stable hook across supported executables.

The source-backed state mapping, movement/combat director behavior, existing
hook inventory, prioritized body-only patch spike, and required I20 fixtures are
frozen in [`i20_native_unit_animation_handoff.md`](i20_native_unit_animation_handoff.md).

## Scroll, Wrap, Clip, And Refresh Invariants

Every production frame or cached fragment must obey all of these rules:

1. **Civ III owns redraw timing.** A timer may set the existing animator dirty
   bit; it may not render, blit, recurse, wait, or pump messages.
2. **Canonical identity and screen occurrence are different keys.** Logical
   state uses canonical `(tile_x, tile_y)`. Placement and ownership use
   `(tile_x, tile_y, anchor_x, anchor_y)` because one wrapped tile can appear at
   multiple screen positions.
3. **Captured content overrides dirty hints.** Dirty flags are optimization
   hints, never proof that a cached result is valid. A frame fingerprint must
   include target size, clip/fragment bounds, zoom/tile basis, ordered tile
   occurrence anchors and renderer-owned state, visible player/visibility,
   world dimensions/wrap, environment, pack/definition revision, ownership
   revision, and device/reset generation. Any mismatch rejects stale cache data
   even if a timer-requested redraw carried only `DIRTY_DYNAMIC`.
4. **Partial repaints are fragments, not complete viewports.** Cache entries
   must record their coverage. A fragment may update only its covered pixels;
   transparent pixels outside it must not erase the retained map. Tall or
   cross-tile geometry must include the same conservative neighbor margin used
   by the accepted full scene.
5. **Camera changes cannot be consumed as animation-only work.** Scrolling,
   zoom, wrap-occurrence changes, resize, visibility changes, and destination
   recreation force scene/composite validation regardless of
   `custom_renderer_redraw_pending`.
6. **No reentrant native draw.** A nested `m71` call is suppressed while custom
   rendering owns the map plane; it can neither append to the outer capture nor
   invoke a hidden vanilla recovery path.
7. **Exclusive full-plane ownership.** An active custom configuration omits the
   whole original `m19` tile plane. Load, capture, render, validation, blit,
   device, missing-system, and reentrant failures are visible hard failures and
   never replay native terrain. Configuration off is the sole vanilla path.
8. **Lifecycle changes invalidate everything.** Scenario/rules changes unload
   the renderer so layered definition paths are resolved again. Device errors
   reset native resources and retry only on a later normal Civ III draw.
9. **Map and unit planes invalidate independently.** Static map caching must not
   prevent unit dirty-region refresh, and unit animation must not require a full
   map rebuild unless Civ III already requests one or renderer-owned static
   content changed.
10. **One UI-thread presentation path.** Both planes render off-screen and copy
    into Civ III-owned surfaces. Neither creates a presenter or competes for the
    game window.

## Required M6.7 Evidence

Automated/replay evidence must cover pixel and sub-tile scroll deltas, both
zooms, horizontal wrap with duplicate logical occurrences, partial clips,
resize/minimize/restore, player/visibility and environment changes, scenario
definition changes, renderer reset, synthetic device loss, rejected partial
output, blit failure, and a forced nested draw. Cache telemetry must distinguish hits,
misses, evictions, and fingerprint-based stale rejections.

The optional final in-game checkpoint should exercise one continuous sequence:
drag/edge scroll, jump-to-unit/city, zoom in/out, cross the wrap seam, expose a
new tile, move and animate a native unit over custom terrain, cover/uncover the
window, and reload a save/scenario. Until M7.4, the expected result is native
units and unit HUD above custom terrain with no trails, duplicate terrain,
missing relief, stale strips, or forced continuous redraw on a static map.
