# City palace asset intake

Status: complete standard-game inventory and offline root conversion; visual
selection and native ownership remain L17 work.

## Result

The earlier two-palace result was caused by a narrow search of the Gran
Colombia/Maya content package. An exact scan of every installed
`Landmarks.artdef` for `Tag_HeroBuilding = Building:BUILDING_PALACE` finds 95
bindings: 48 standard-game bindings, 47 scenario-only bindings, and 47 distinct
standard-game package/entry roots. Every standard binding resolves to an
installed package. Scenario variants are reported but deliberately excluded
from the general library.

`palace_asset_importer.py` converts all 47 standard roots into the generic
`CityPalacesNormalized` local pack. The current build contains 585 geometry
parts, 464 materials, 124 emissive material bindings, 388 unique textures, and
257 attachment points. Seven all-empty source-material draws across five
palaces are treated as invisible sentinel draws and omitted; no replacement
color is invented.

One limitation is preserved explicitly. The Gran Colombian root has four
permanent `Tree_B_Lg` attachments whose package records are not landmark/city
block records, so the current compound decoder cannot normalize those four
trees. The palace body itself converts. The catalog therefore reports
`root_library_normalized_with_unresolved_required_attachments`; L17 must not
claim that style as complete until the tree record class is decoded or the
profile deliberately substitutes a generic foliage child.

## Installed standard roster

The ArtDef selectors below are source evidence, not runtime civilization IDs.

- Base regional or generic roots: Ancient Brick, Ancient Earth, Ancient Wood,
  American, Brazilian, Southeast Asian, default, East Asian, Mediterranean,
  Mughal, North African, South American, South African, Baltic, Indonesian,
  and Scottish.
- Base civilization roots: German, Norwegian, English, Spanish, and Sumerian.
- DLC roots: Australian, Babylonian, Byzantine, Gaulish, Ethiopian, Cree,
  Korean, Mapuche, Mongolian, Dutch, Zulu, Canadian, Hungarian, Incan, Malian,
  Maori, Ottoman, Phoenician, Swedish, Gran Colombian, Mayan, Vietnamese,
  Nubian, Polish, and Portuguese.
- Expansion 2 also binds a separate `DIS_CTY_CREE_Palace` package to Cahokia.
  It has the same source entry name as the Expansion 1 Cree palace but is a
  distinct package payload and receives a distinct generic asset ID.

The source roots are `DIS_CTY_AB_Palace`, `AE`, `AW`, `RAM`, `RBRZ`, `RC`,
`RE`, `RGER`, `RJ`, `RMED`, `RMUG`, `RNA`, `RNWY`, `RSA`, `RSS`, `RBAL`,
`RENG`, `RIND`, `RSCT`, `RSPN`, `RSUM`, `RAU`, `RBAB`, `RBYZ`, `RGAU`,
`RETH`, `CREE` (two packages), `RKOR`, `RMAP`, `RMON`, `RNTH`, `Zulu`,
`RCAN`, `RHUN`, `RINC`, `RMAL`, `RMAO`, `ROTT`, `RPHO`, `RSWD`, `RCOL`,
`RSAM`, `RVIE`, `RNUB`, `RPOL`, and `RPOR`, each with the common
`DIS_CTY_`/`_Palace` spelling where applicable. The generated probe report
retains exact package paths, source entry spelling, ArtDef files, selectors,
and resolution evidence without putting source paths into the runtime pack.

## Runtime selection contract

Capital visibility comes only from authoritative Civ III `is_capital` state.
The selected palace is an additive center accent; it never determines capital
status and never replaces the rest of the city composition.

The asset pack exposes generic palace IDs. A modder-owned city-style profile
selects them in this order:

1. explicit scenario/civilization city-style override;
2. Civ III culture-group fallback;
3. pack default.

The source culture/civilization tags remain provenance only. Renderer code
must not test Civ VI civilization IDs. Era changes retain the chosen palace
style unless the active pack explicitly supplies an era override; the three
ancient source variants are candidates for authored ancient profiles, not an
automatic excuse to switch a civilization to an unrelated later palace.

If a selected style is missing or incomplete, render the ordinary city and
retain Civ III's native capital indicator. This allows arbitrary scenarios and
partial mod packs to remain correct.

## Reproduction

From the project root:

```bash
python3 Renderer/tools/asset_compiler/palace_asset_probe.py
python3 Renderer/tools/asset_compiler/palace_asset_importer.py
PYTHONPATH=. python3 -m unittest \
  Renderer.tools.asset_compiler.test_palace_asset_importer
```

The normalized pack and reports are local ignored derivatives and are not
redistributed. This work does not enable native rendering and does not advance
the active Lab gate. No Civ III patch symbol is needed.
