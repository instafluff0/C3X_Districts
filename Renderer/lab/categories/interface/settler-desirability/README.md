# Settler city-site desirability

`python3 Renderer/renderer.py lab settler-desirability` makes three visual
examples: an outlined translucent tile gradient, a coastal version with omitted
ineligible tiles, and a plain wash for comparison. Each runs at tile widths 128,
160 and 192 over a current production-rendered scene. These are the custom
renderer zoom widths in `advance_custom_renderer_zoom_from_key`. The fixture's
tile anchors follow `biq_preview.cpp`: X advances by half the tile width per
raw tile coordinate, Y by half the tile height, and tile height is half width.
The overlay itself is a Lab mockup, not a game or DLL feature.

The invented evaluation values are graded with C3X's current 11-sprite formula:
positive values near 1,000,000 span white (least desirable) to deep green (most
desirable). Evaluation zero is never drawn. The fake scores are a visual fixture,
not a claim about Civ III's AI or city-placement rules. In game, C3X's
`patch_Match_ai_eval_city_location` and current display perspective must remain
the authoritative source of both eligibility and grade.

The old `Art/TileHighlights.pcx` contains eleven 128×64 outlined sprites.
`init_tile_highlights` slices those exact dimensions, and the city-site branch
passes the tile hook's `pixel_x, pixel_y` to `patch_Sprite_draw_on_map`.
That branch suppresses the sprites when Civ III's native zoom-out flag is set;
the custom renderer's 128/160/192 frame instead scales captured tile anchors.
The filled Lab treatment intentionally changes the old outline-only appearance.

The production handoff later needs a renderer-owned layer with copied tile
coordinates, evaluation values, visibility, and redraw lifetime. Its custom-on
path should skip only the old city-site sprite draw; other tile highlights need
their own ownership decision. Configuration-off keeps the existing sprite path.
