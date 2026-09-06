# C3X To Civ VI Natural-Wonder Seed Mapping

`Renderer/inventory/vanilla_c3x_to_civ6_natural_wonders.json` is the editable first-pass mapping for all 18 definitions in `default.districts_natural_wonders_config.txt`. It is conversion input and planning evidence, not a hard-coded runtime table.

The ledger preserves each default definition's order, name, terrain and adjacency requirements, native PCX cell, and configured animation count. It also records a preferred installed Civ VI `FEATURE_*` source, alternatives, confidence, expected compiled-kit parts, orientation policy, and complete native fallback. The validator fails when the default config and ledger drift apart.

## Current Coverage

| Result | Count | Definitions |
| --- | ---: | --- |
| Exact named source | 8 | Yosemite, Mount Everest, Mount Kilimanjaro, Great Barrier Reef, Matterhorn, Eyjafjallajokull, Ha Long Bay, Delicate Arch |
| Approximate source | 9 | Angel Falls, Mount Fuji, Yellowstone, Zhangjiajie Mountains, Moraine Lake, Tropical Rainforest, Wadi Rum, Lofoten Skerries, Geirangerfjord |
| Authored art required | 1 | Savanna |

Approximate means source material worth testing, not visual approval. In particular, Mount Fuji, Tropical Rainforest, and Wadi Rum are deliberately low-confidence placeholders. Their definitions must remain easy to override after contact-sheet or in-game curation. Savanna has no defensible installed natural-wonder match and therefore keeps native C3X art until a kit is authored.

## Source Resolution

Run the portable validation:

```powershell
py Renderer\inventory\natural_wonder_mapping_inventory.py
```

Resolve every non-authored preferred target against an installed Civ VI tree:

```powershell
py Renderer\inventory\natural_wonder_mapping_inventory.py `
  --civ6-assets-root "Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets" `
  --require-all-targets
```

Resolution scans only `Features.artdef` definitions, rather than accepting incidental references in VFX or selector files. It reports Base/DLC provenance without copying cooked payloads. A later source probe still has to establish the model pieces, footprint, transforms, water integration, VFX, emissives, and light attachments for each choice.

## Runtime Rules

- Resolve by authoritative C3X `natural_wonder_id` and effective scenario/user/default config identity.
- Compile source data into a source-agnostic natural-wonder kit; runtime code never sees `FEATURE_*`, ArtDef, or BLP identifiers.
- Root every kit at the one authoritative C3X tile. Civ VI multi-tile pieces may overhang visually but cannot create gameplay tiles.
- Use C3X adjacency direction first where one is configured, then a deterministic source best fit.
- Replace the body exclusively. Missing geometry, directional pieces, water support, or required attachments fail the custom pass visibly without native PCX or animation replay.
- Keep native fog/shroud, natural-wonder label, minimap, HUD, UI, and gameplay state unchanged.

Scenario and user natural-wonder configs can replace the full default roster and C3X supports up to 64 definitions. M9 must therefore generate or accept mappings for the effective roster and use this 18-entry file only as the vanilla seed. Unknown scenario identities require an explicitly authored mapping before their custom-on integration can pass.
