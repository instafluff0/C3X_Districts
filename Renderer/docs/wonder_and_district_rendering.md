# Wonder And District Rendering Contract

## Scope And Sequence

Constructed wonders and C3X districts are first-class renderer categories, but they follow the general map-object, animation, authoring, and validation work. They are deliberately separate:

1. **M10 constructed wonders:** inventory every BIQ Great and Small Wonder, correlate the effective C3X wonder-district configuration, seed Civ VI/source mappings, and render construction/completed wonder models when they have a map tile.
2. **M11 districts:** inventory every effective C3X district definition and runtime state, map district bases and building additions, then render composite district kits. M11 is the final planned renderer milestone.

Natural wonders are a separate M9 category governed by `natural_wonder_rendering.md`. M11 inventories the reserved Natural Wonder district type only to prove that it delegates to M9 and is neither duplicated nor omitted.

## Initial Configuration Census

The checked-in defaults currently contain:

- 23 `#Wonder` entries in `default.districts_wonders_config.txt`.
- 21 `#District` entries in `default.districts_config.txt`.
- 11 ordinary configurable definitions: Encampment, Holy Site, Campus, Entertainment Complex, Commercial Hub, Industrial Zone, Data Center, Offshore Extraction Zone, Park, Ski Resort, and Water Park.
- 10 special definitions: Neighborhood, Wonder District, Distribution Hub, Aerodrome, Port, Central Rail Hub, Energy Grid, Bridge, Canal, and Great Wall.

This is only a seed census. The final inventories parse the effective default, user, and scenario files using C3X precedence, enumerate dynamic definitions, and correlate names to loaded BIQ IDs. A filename or `#District`/`#Wonder` block count cannot satisfy either milestone.

The checked-in first-pass ledgers are `Renderer/inventory/vanilla_conquests_wonder_roster.json`, `Renderer/inventory/vanilla_c3x_to_civ6_constructed_wonders.json`, and `Renderer/inventory/vanilla_c3x_to_civ6_districts.json`. Their shared validator is `Renderer/inventory/wonder_district_mapping_inventory.py`, and curation policy is documented in `Renderer/docs/civ3_to_civ6_wonder_and_district_mapping.md`. The BIQ roster classifies all 40 vanilla Great/Small Wonders, including the 17 that remain mapless without a default C3X wonder definition. M10/M11 must still inventory each effective scenario/user roster and placed runtime instance.

## Constructed Wonder Contract

Each effective wonder record must capture:

- Stable BIQ improvement ID and name, Great/Small Wonder class, owner, and visibility.
- Whether it has a C3X Wonder District tile and that tile's stable instance ID, coordinates, authoritative screen anchor, terrain, river/city adjacency, and orientation.
- Planned, under-construction, completed, destroyed, abandoned, obsolete, and hidden states that can affect drawing.
- Normal and alternate-direction source sprite metadata, construction/completed selection, width/height, offsets, and evidence.
- Source-agnostic replacement asset IDs, calibration, footprint, season/day-night behavior, and fallback ownership.

The first mapping seed should correlate Civ III names to Civ VI Wonder/Building ArtDefs where possible and explicitly classify absent, renamed, or anachronistic matches. A Civ III wonder with no map tile remains inventoried as `not_map_rendered`; the renderer must not invent placement. Scenario mappings can choose a different model without changing importer code.

## District Instance Contract

Every visible district instance records:

- Stable district instance ID, tile coordinates and anchor, district numeric ID/name/type, owner, culture, era, and deterministic variant seed.
- Construction/completion, pillaged/destroyed/abandoned, obsolete, visibility/fog, resource overlap/subsumption, and source sprite selection.
- `render_strategy`, effective building count, and the stable BIQ IDs/names of each dependent building currently represented at that tile. For shared districts, this is the effective visual set produced by C3X's in-range city rules, not a guessed owner city.
- Terrain and coastline alignment plus directional/topology masks required by Port, Bridge, Canal, Great Wall, and similar scenario-defined types.
- Source-agnostic base, attachment, preset, construction, damage, and topology asset IDs with calibration and ownership.

Gameplay remains authoritative in C3X. The renderer observes district and building state; it never grants a building, chooses which city shares it, completes construction, changes passability, or modifies yields.

## Composite District Kits

A 3D district is a logical kit rather than one mandatory model:

```text
DistrictKit
  base
  construction/damaged/abandoned alternatives
  building attachments keyed by stable Civ III building ID/name
  count-stage presets
  deterministic sockets/transforms
  optional directional/topology pieces
```

This separates C3X semantics from Civ VI file organization. A source adapter may obtain the base and attachments from Civ VI District, Building, Improvement, Route, or Landmark ArtDefs, but the compiled pack contains only generic models/materials and stable C3X logical IDs.

### `by-building`

The base model is always present. Each dependent building that C3X says is visually present adds its independently mapped attachment. Attachments use authored sockets/transforms when available; otherwise the pack compiler assigns deterministic collision-aware positions from the district footprint. Placement is stable across redraws, save/load, seasons, and machines.

This is the direct 3D analogue of the current base sprite plus one separately drawn PCX column per completed dependent building. It also allows a later configuration to swap only `Campus.Library`, for example, without rebuilding or replacing the Campus base.

The same component owns the presentation details it introduces. A Library attachment may own lit windows; a Factory may own windows, furnace glow, smoke, and sparks; a district base may own common lamps or a brazier. Removing a represented building removes only its lights/effects. Custom-frame failure removes the complete renderer-owned light/effect set so nothing floats after the body fails. It never restores the native district while custom rendering is enabled. All activation and animation use the shared contract in [environment_lighting_and_ambient_effects.md](environment_lighting_and_ambient_effects.md), not a district-specific clock.

### `by-count`

The captured effective building count selects a named stage preset. A preset may reference a monolithic staged model or a deterministic attachment set; it does not need to preserve which individual building produced that count because the current C3X strategy does not expose that distinction visually.

### Ownership failure

Replacement is complete per district instance. The I# gate does not transfer the
2D district until the complete base plus every required visible attachment,
preset, and topology piece is ready. After transfer, any missing mapping, failed
load, or unsupported state fails visibly without restoring native C3X district
art. Mixed 2D/3D body composition is not a fallback mode.

## Special Families

- Wonder District delegates its constructed centerpiece to M10 while M11 owns the district pad and relationship.
- Bridge, Canal, and Great Wall require connection/direction masks and piece ordering rather than building-count stages.
- Port requires authoritative coastline orientation and may use alternate directional assets.
- Neighborhood and other decorative stages require deterministic variants independent of frame time.
- Construction, damage, pillage, and abandonment remain distinct states even when the source pack initially falls back to a shared placeholder.

## Gates

M10 is complete only when every BIQ Great/Small Wonder and effective wonder config entry is classified, mapping coverage is reported, construction/completed orientation fixtures pass, mapless wonders remain mapless, and missing assets preserve C3X rendering.

M11 is complete only when every effective default/user/scenario district definition and every runtime render strategy/state is inventoried; `by-count`, `by-building`, shared-district additions, construction, abandonment, culture, era, coast alignment, topology, component-owned emissive, and ambient-effect families have deterministic fixtures; scenario overrides work; configuration off preserves current C3X district visuals; and custom-on failure never substitutes them.

Manual screenshots follow the normal strategic checkpoint budget. Contact sheets, synthetic district kits, replay scenes, and deterministic state matrices are the ordinary iteration evidence.
