# C3X To Civ VI Wonder And District Seed Mapping

These editable conversion ledgers give M10 and M11 a tested starting point:

- `Renderer/inventory/vanilla_conquests_wonder_roster.json` classifies all 40 vanilla Conquests Great/Small Wonders directly from BIQ flags: 23 have default C3X mappings and 17 remain mapless.
- `Renderer/inventory/vanilla_c3x_to_civ6_constructed_wonders.json` covers all 23 default C3X constructed-wonder definitions and correlates each one with its vanilla Conquests BIQ identity.
- `Renderer/inventory/vanilla_c3x_to_civ6_districts.json` covers all 21 default C3X districts, their current map-art metadata, and all 22 dependent-building roles.
- `Renderer/inventory/wonder_district_mapping_inventory.py` validates both ledgers against the checked-in configs and BIQ semantics, then optionally resolves every non-authored source target against installed Civ VI ArtDefs.

These are source-conversion choices, not runtime identity tables. Effective scenario or user configs replace the default rosters, and unknown identities retain complete native C3X rendering until explicitly mapped.

## Constructed Wonders

The configured seed contains 11 exact source identities, 11 approximations, and one authored requirement: the Pentagon. The 17 mapless BIQ wonders also have dormant source hints for future scenario configs; the Manhattan Project is the sole additional authored requirement. Approximate sources include deliberate compositions such as Hoover Dam from Civ VI's Dam plus Hydroelectric Dam, and Apollo/SETI from Spaceport parts. They require contact-sheet curation and calibrated construction/completed states before M10 can suppress native art.

Every record preserves BIQ index and Civilopedia key, Great/Small Wonder class, C3X placement restrictions, construction and completed PCX cells, and alternate-direction cells. The BIQ snapshot generator derives class from `BLDG.otherChar` bits 2 and 3. A BIQ wonder is not placed on the map merely because it has a mapping or dormant source hint. It becomes renderer-eligible only when C3X supplies an effective wonder definition and authoritative Wonder District instance.

## Districts And Building Additions

Every district has a separately mapped base and an ordered attachment record for each `dependent_improvs` entry. The checked-in defaults currently use `by-count`, so the complete effective building count selects a stage preset. The same attachment records become independent components when a scenario chooses `by-building`:

```text
Campus base -> DISTRICT_CAMPUS
Library attachment -> BUILDING_LIBRARY
University attachment -> BUILDING_UNIVERSITY
```

This lets a scenario replace one attachment without changing the base. It also establishes ownership for component-specific night windows, lamps, smoke, flames, and other ambient effects. If any required visible component is unavailable, the complete native district draw returns; partial mixtures are not implicit.

Special families remain explicit:

- Wonder District supplies a base/pad but delegates the constructed centerpiece to M10.
- Port and offshore districts retain coast alignment and water behavior.
- Canal, Bridge, and Great Wall require connection topology, not ordinary building stages.
- Central Rail Hub and Bridge currently require authored kits.
- SAM Missile Battery, Mass Transit System, and Nuclear Plant are explicit authored attachment gaps.

## Validation

Portable config and ledger validation:

```powershell
py Renderer\inventory\wonder_district_mapping_inventory.py
```

Installed Civ VI resolution:

```powershell
py Renderer\inventory\wonder_district_mapping_inventory.py `
  --civ6-assets-root "Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets" `
  --require-all-targets
```

Resolution accepts definitions only from `Buildings*.artdef`, `Districts*.artdef`, and `Improvements*.artdef`. It reports Base/DLC provenance without extracting cooked assets. M10/M11 still need source-piece, transform, footprint, construction, damage, emissive, VFX, and light probes before any mapping becomes a complete runtime kit.
