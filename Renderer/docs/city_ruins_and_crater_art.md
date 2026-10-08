# Ruins, craters, pollution and lava damage

Status (2026-10-07): the user chose ash and char for all pollution, tuned blast
craters, and a dark rubble field for ruins, and asked for them in the game. The
rule is to follow Civ III's tile state exactly: no native art is hidden.
The implementation is described under *Integration* below.

## Current gap

Custom rendering skips Civ III's whole m19 tile draw
(`patch_Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`). City ruins, craters and
pollution therefore do not appear on the custom map. Capture already copies
the pollution and crater bits (`capture_custom_renderer_overlay_body`). Ruins are
not captured. The same skip hides forts, barricades, colonies, airfields, radar
towers, outposts and victory points; see
[remaining tile infrastructure](remaining_tile_infrastructure.md).

## Native semantics (confirmed from the GOG decompile unless marked)

City ruins:
- Stored in `Tile::Ruins` (+0x24). The accessors are `m36_Get_Ruins` and
  `m60_Set_Ruins`. Neither takes a viewer, so ruins show the global state;
  only the separate fog pass masks unexplored tiles.
- Set by `City::raze`, which every city removal reaches: abandonment, razing
  after capture, the capture destroy branch, elimination, and a volcano
  eruption onto a city. Starvation and nukes shrink a city but never raze it.
  C3X sets ruins when an attack destroys a land district.
- Cleared by the worker jobs mine, irrigate, fortress, road, rail, plant
  forest, airfield, radar, outpost and barricade, and when a hut or camp is
  placed. `Leader::create_city` shows no ruins write (inferred from the
  decompile). The user's experience is that founding a city removes them, which
  is unverified here. The renderer follows the flag either way.
- Drawn first in `m12_Draw_Tile_Buildings`. The game picks one of the three
  `Art/Cities/DESTROY.PCX` frames (167x95 each) at random, seeded by the tile,
  not by the city's size. Their content is 84x45, 98x54 and 133x70 px:
  town-to-city scale, about twice a goody hut.

Pollution (overlay bit 0x40):
- `m20_Check_Pollution(viewer)` reads `m42_Get_Overlays(viewer)`. That
  returns the viewer's remembered low byte while the tile is out of sight,
  so pollution is remembered under fog and can be stale.
- Volcano eruptions set the same bit. `erupt_onto_tile` runs on the volcano
  itself and on up to three of its eight neighbours. On land it removes routes
  and improvements, sets pollution, razes any city (leaving ruins) and kills
  units. C3X's `patch_Tile_set_flag_for_eruption_damage` hooks that call.
- Nothing stored distinguishes eruption pollution from city, meltdown or nuke
  pollution. A renderer can only infer it: pollution on a volcano or one of
  its eight neighbours. Industrial pollution next to a volcano would then also
  read as lava.

Craters (bit 0x100, inferred from writers; the m21 body is not decompiled):
- Created by a successful bombard from a unit type with `Create_Craters`, once
  the tile has no improvement tier left to strip. Nukes pollute instead.
- Removed by Clean Pollution, but only after the tile's pollution is gone.
- Bit 0x100 lies outside the remembered byte, so natively craters vanish under
  fog. `test_injected_world_authority.py` mocks the crater check as bit 4 (the
  fortress bit) and expects it to be remembered. Fix that mock when craters
  get rendered.

Drawing both: `m13_Draw_Tile_Pollution` draws craters, then pollution. Each
uses a 5x5 sheet of 128x64 cells. An isolated tile picks one of variants 0-9 at
random; a tile with same-state diagonal neighbours uses 9 + mask (NW=1, NE=2,
SE=4, SW=8). Note that this order is clockwise, unlike the irrigation mask.

## Existing Civ VI candidates

- Craters: four `CLUTTER_CRATERBLASTS` decals, normalized as
  `infrastructure/crater/variant_01..04` in the local `FutureGateCandidates` pack.
- Ruins: `TEXTURE_Ruin_Debris_Decal`, its `_H` auxiliary and a lighter
  variant, all from Base `landmarks/tilebases.blp`.
  `city_ruin_texture_importer.py` converts them into the local
  `CityRuinsCandidates` pack. Package evidence does not prove that Civ VI uses
  this decal for a razed city.
- Pollution: an older intake chose `NUCLEAR_FALLOUT -> FX_Radiation` (green).
  `build_infrastructure_runtime.py` writes it into a `ground_state_runtime.bin`
  that no runtime loads. That look does not suit lava damage.

Survey of 2026-10-07 (the ArtDef chains are confirmed; the suitability verdicts are inferred):
- Lava damage: Expansion2 `FEATURE_VOLCANIC_SOIL` -> `CLUTTER_VOLCANIC_SOIL` ->
  `Feature_VolcanicSoil_Decal_001-004`, texture `TEXTURE_Feature_VolcanicSoil_B/H/S`.
  Gran Colombia/Maya burnt-forest ground decals: `TEXTURE_Forest_BurnDecal_B` and
  `TEXTURE_Forest_Burn_Variant_B`. Expansion2 `TEXTURE_DiffuseTint_ScrollingMagma_B_null`.
- Craters: Gran Colombia/Maya `RES_Meteor_Debris`, decal 0, texture
  `TEXTURE_Decal_Craters_B/H`. Civ Royale `PIL_CraterDecal001-004`, one quarter each of
  `TEXTURE_Decals_Crater_Blast_B/H`. Civ Royale `PIL_BlastDecal01-04`. Base `TEXTURE_FX_ScorchMark`.
- Ruins: `RESOURCE_ANTIQUITY_SITE` -> `CLUTTER_ANTIQUITY_SITE` -> `IMP_Antiquity_*`
  meshes in Base `environment/clutter.blp`. They import with
  `compound_landmark_importer`. Base `TEXTURE_Burn_BaseColor` is the char map.
- Importer bug: `generic_decal_compiler.decode_decal_mesh` reads decal quads through
  the wrong index buffer. Reading vertices `first..first+count` directly gives exact
  quads. The `FutureGateCandidates` crater decals carry no mesh, so each would draw
  the whole 2x2 atlas.

Draft mockups (CPU composites on the 1498 AD frame) are in
`Renderer/lab/out/ground_states_mockup/index.html`. They also show that huts and camps
are stretched 2.6x vertically by `build_site_runtime.py`'s `HEIGHT_SCALE`; mines are not.

Derived Civ VI pixels stay local and are not redistributed.

## Integration

1. Capture: `C3X_RENDERER_IMPROVEMENT_RUINS = 128u`, read with `m36_Get_Ruins`
   beside the tile-building capture (see the patch ledger). Pollution and craters
   were already captured.
2. Pack:
   - `ground_state_asset_importer.py` (mapping `ground_state_sources.json`)
     imports the sources into the ignored `GroundStatesNormalized` pack.
   - `build_site_runtime.py` with `ground_state_composer.py` bakes each look into
     `TileSitesRuntime`. The looks are data in `tile_site_looks.json`.
   - Groups: `pollution_0..3` and `crater_0..3` (one draped decal each, sharing a
     2x2 atlas), and `ruins_0..2` (one rubble decal, at Civ III's three sizes).
3. Runtime: `select_improvements` adds them after huts and camps, in the order
   pollution, craters, ruins.
   - Decals are `decal/ground/*` assets with owner code 0, so they take the
     resource-decal material: soft alpha and the natural depth basis.
   - The steep-decal skip and farm clipping do not apply to them. Water
     tiles carry none.
   - Ruins follow the tile's flag alone, with no city special case.
   - Decals lie at a lift of .02, over crop fields (.016) and under route
     strips. The site layer draws after farms, as Civ III draws pollution over
     irrigation; crops are never hidden.
   - Crater and rubble relief is baked sunlit, so they never turn. Pollution
     turns freely.
4. Lab: `Renderer/lab/studies/ground_states/study.py` replays the 1498 AD save
   around the big volcano. `C3X_LAB_TILE_OVERLAYS` maps pollution (0x40) and
   crater (0x100) bits, and `C3X_LAB_TILE_RUINS` lists ruin tiles.
