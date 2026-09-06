# M6.0 Vanilla Map-Art Inventory Evidence

M6.0 is complete without a manual game run. The strict local gate reads the installed vanilla Civilization III Complete/Conquests files, regenerates BIQ semantics through the existing read-only C3X Editor parser, inventories layered art, validates every explicit Civ III atlas slice, correlates unit selectors with INI/FLC metadata, and generates annotated PNG contact sheets in a temporary directory.

## Closed census

- 112 effective map-related art files across Base, Play the World, and Conquests search precedence.
- 76 effective PCX atlases, each with explicit slice rectangles, authored capacity, reachable indices, and a generated annotated contact sheet.
- 124 primary Conquests BIQ unit types. The other 17 PRTO records are AI-strategy child rows, not additional unit types.
- 144 selectable unit-art directories: 134 reachable through effective `ANIMNAME` selectors and 10 explicitly unreachable in standard Conquests rules.
- 26 BIQ resources and their exact `GOOD.icon` reachability over 36 authored resource cells.
- 14 BIQ terrain types, five city culture groups, four eras, three normal city sizes, and the separately sliced walled/destroyed/status states.
- 21 renderer/retained/editor responsibility families, all classified and tested with zero unknown selectors.

FLC records preserve the embedded Civ III `num_anims`, frames-per-direction, SW/S/SE/E/NE/N/NW/W order, smoke/shadow palette ramps, and transparency index. One- and two-direction missile, nuclear, and paradrop/build clips are recorded as format-declared exceptions instead of being incorrectly promoted to eight directions.

## Reproduction

```powershell
node Renderer\inventory\extract_biq_semantics.js --biq "..\conquests.biq" --editor-root "..\C3X_Editor" --install-root "..\.." --output Renderer\inventory\vanilla_conquests_biq_semantics.json

py Renderer\inventory\civ3_art_inventory.py `
  --install-root "..\.." `
  --output Renderer\inventory\generated\vanilla_conquests_art.json `
  --markdown Renderer\inventory\generated\vanilla_conquests_art.md `
  --contact-sheets Renderer\inventory\generated\contact_sheets `
  --fail-on-unclassified --fail-on-unresolved

py Renderer\tools\verify_project.py --require-local-assets
```

Generated contact sheets and Firaxis-derived pixels remain ignored and local. The tracked BIQ snapshot contains identifiers, indices, hashes, and selector metadata only; it does not contain the BIQ or any art payload. Districts are excluded.
