# Settler city-site desirability

`python3 Renderer/renderer.py lab settler-desirability` makes three visual
examples: an inset translucent tile gradient, a coastal version with omitted
ineligible tiles, and a plain wash for comparison. Each runs at tile widths 128,
160 and 192 over a current production-rendered scene. These are representative
Lab widths; Renderer64 captures at 128 and presents continuous zoom from 1× to
3×. The fixture's
tile anchors follow `biq_preview.cpp`: X advances by half the tile width per
raw tile coordinate, Y by half the tile height, and tile height is half width.
The Lab values are mockups; the production renderer uses C3X's live scores.

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
The inset Lab treatment intentionally changes the old outline-only appearance.

The production path uses C3X's active perspective and evaluation on the game
thread, copies grades with tile occurrences, and draws the same green palette
on the final GPU map before fog. The original custom-on hook already bypasses
all legacy PCX highlights; configuration-off keeps their original behavior.
District worker and focus highlights remain separate future ownership work.
The palette uses a cubic white-to-green ramp so neighboring top grades remain
distinguishable through the translucent fill. Each colored diamond stops one
pixel inside its tile edge, leaving a narrow terrain gap without an outline.
The Lab's invented values do not establish the distribution of live scores. A bounded early-game save yielded
12 visible legal sites, with three in grade 9 and nine in grade 10; C3X's raw
evaluations ranged from 1,000,038 to 1,000,057. The focused live test showed
the overlay following both automatic settler selection and the L-key picker.
