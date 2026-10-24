# C3X To Civ VI Natural-Wonder Seed Mapping

`Renderer/inventory/vanilla_c3x_to_civ6_natural_wonders.json` is the editable first-pass mapping for all 18 definitions in `default.districts_natural_wonders_config.txt`. It is conversion input and planning evidence, not a hard-coded runtime table.

The ledger preserves each default definition's order, name, terrain and adjacency requirements, native PCX cell, and configured animation count. It also records the chosen `FEATURE_*` source, where that source comes from (`source`), alternatives, confidence, expected compiled-kit parts, the kit-owned parts each season needs (`season_parts`), orientation policy, and complete native fallback. The validator fails when the default config and ledger drift apart.

## Current Coverage

The user approved the Lab look of all 18 kits, in all four seasons, on 2026-10-23 (`Renderer/lab/studies/natural_wonders/kits.py`). They are not integrated yet.

| Result | Count | Definitions |
| --- | ---: | --- |
| Exact, base game | 8 | Yosemite, Mount Everest, Mount Kilimanjaro, Great Barrier Reef, Matterhorn, Eyjafjallajokull, Ha Long Bay, Delicate Arch |
| Exact, Workshop | 2 | Angel Falls (Sukritact's Angel Falls), Zhangjiajie Mountains (Terra Mirabilis Wulingyuan) |
| Approximate | 4 | Yellowstone (Terra Mirabilis Old Faithful), Moraine Lake (Crater Lake with the Matterhorn), Wadi Rum (Sukritact's Monument Valley), Geirangerfjord (Lysefjorden) |
| Authored kit | 4 | Mount Fuji (stratovolcano), Tropical Rainforest (jungle canopy), Lofoten Skerries (Matterhorn-peak islets), Savanna (JNR Savannah models) |

Source kinds:

- `base`: installed Civ VI and its DLC.
- `workshop`: a named Steam Workshop item (Civ VI app 289070). The ledger names the item ID and title.
- `authored`: a kit composed in the Lab, optionally from named source parts.

All three kinds are local conversion input. Like Civ VI assets, Workshop content is never redistributed with C3X or read at runtime. The asset compiler turns every kind into a generic C3X kit.

`season_parts` lists what a kit must ship beyond the shared production season code:

- `autumn_spring_crowns`: kit broadleaf textures recoloured for autumn and spring. Composition bodies get no crown recolour from the season code.
- `winter_snowy_pines`: the kit's pines swapped for the snowy pine meshes.
- `winter_snow_free_basin`: geothermal ground that stays bare in winter.
- `winter_waterline`: no snow at or below the waterline (reef coral).

## Source Resolution

Run the portable validation:

```powershell
py Renderer\inventory\natural_wonder_mapping_inventory.py
```

Resolve every source feature against an installed Civ VI tree and the Workshop folder. Base features resolve in the game. Workshop features resolve only inside their own item folder.

```powershell
py Renderer\inventory\natural_wonder_mapping_inventory.py `
  --civ6-assets-root "Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets" `
  --workshop-root "Z:\Library\Application Support\Steam\steamapps\workshop\content\289070" `
  --require-all-targets
```

On the Mac, use the same folders under `$HOME/Library/Application Support/Steam/steamapps`. On 2026-10-23 all 18 entries resolved.

Resolution scans only `Features.artdef` definitions, rather than accepting incidental references in VFX or selector files. It reports Base/DLC provenance without copying cooked payloads. A later source probe still has to establish the model pieces, footprint, transforms, water integration, VFX, emissives, and light attachments for each choice.

## Runtime Rules

- Resolve by authoritative C3X `natural_wonder_id` and effective scenario/user/default config identity.
- Compile source data into a source-agnostic natural-wonder kit; runtime code never sees `FEATURE_*`, ArtDef, or BLP identifiers.
- Root every kit at the one authoritative C3X tile. Civ VI multi-tile pieces may overhang visually but cannot create gameplay tiles.
- Use C3X adjacency direction first where one is configured, then a deterministic source best fit.
- Replace the body exclusively. Missing geometry, directional pieces, water support, or required attachments fail the custom pass visibly without native PCX or animation replay.
- Keep native fog/shroud, minimap, HUD, UI, and gameplay state unchanged. The natural-wonder name is renderer-drawn map text under the fog and unexplored overlay (`docs/natural_wonder_rendering.md`, Labels).

Scenario and user natural-wonder configs can replace the full default roster and C3X supports up to 64 definitions. M9 must therefore generate or accept mappings for the effective roster and use this 18-entry file only as the vanilla seed. Unknown scenario identities require an explicitly authored mapping before their custom-on integration can pass.
