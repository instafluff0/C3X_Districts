# Vanilla Civ III Map Art Inventory Contract

## Purpose

M6.0 establishes the complete replacement and fallback surface for vanilla Civilization III Complete/Conquests before production terrain and map-object packs are claimed complete. C3X natural wonders and Districts are deferred to M9 and final renderer milestone M11 rather than silently excluded from the project. The inventory covers everything visible in or over the vanilla map, including layers that remain owned by Civ III.

The completed inventory has two complementary sources of truth:

1. An offline census of layered Base, Play the World, Conquests, and scenario art roots.
2. BIQ and runtime evidence proving which semantic objects, atlas cells, sprite indices, directions, actions, and overlays Civ III can actually select.

A directory or file count is never treated as a sprite count.

## Scope

Replacement candidates include:

- Flat terrain, terrain transitions, landmark terrain, coasts, sea, ocean, rivers, deltas, waterfalls, and ice.
- Hills, mountains, volcanoes, forests, jungles, marshes, flood plains, snow and other terrain features.
- Roads, railroads, irrigation, mines, fortresses, barricades, colonies, airfields, outposts, radar towers, pollution, craters, huts, victory locations, and other tile-bound infrastructure.
- Strategic, luxury, and bonus resources plus their shadow/visibility behavior.
- Cities by culture group, era, size, walls, capital state, and every map-visible building or wonder distinction Civ III actually draws.
- Every vanilla Conquests unit type and its action, direction, animation, shadow, projectile, and effect bindings.
- Map-visible transient effects including bombardment, impacts, explosions, nuclear effects, disorder, plague, eruptions, smoke, and worker actions where applicable.

Explicitly inventoried retained or separately owned layers include:

- Fog of war and unexplored shroud.
- Territory and cultural borders.
- Map grid, selection highlights, movement/path indicators, targeting markers, and cursor state.
- City labels, population/status marks, unit health/status/selection overlays, and map text.
- Minimap, HUD, advisors, menus, and other UI surfaces that establish the map compositing boundary.
- Editor-only markers, recorded as out of runtime scope instead of silently ignored.

City-screen art, Civilopedia art, leaderheads, diplomacy, wonder splashes, movies, and sounds are not renderer replacement targets unless runtime evidence proves they participate in the map view. City improvements and wonders do not automatically receive 3D assets merely because they exist in the BIQ; the inventory records whether Civ III gives them an individual map-visible representation.

## Generated Ledger

Run the source census from the C3X root:

```powershell
py Renderer\inventory\civ3_art_inventory.py `
  --install-root "C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete" `
  --output Renderer\inventory\generated\vanilla_conquests_art.json `
  --markdown Renderer\inventory\generated\vanilla_conquests_art.md `
  --contact-sheets Renderer\inventory\generated\contact_sheets `
  --fail-on-unclassified --fail-on-unresolved
```

Optional `--scenario-art-root` arguments are appended in increasing precedence. Output paths and asset identities are relative; machine-specific install paths are not written into the ledger.

The `c3x.civ3_art_inventory.v0` ledger records:

- Every relevant source candidate and its effective layered winner.
- PCX dimensions, explicit Civ III slice rectangles, authored capacity, reachable/unreachable indices, and annotated local contact sheets.
- Unit INI action bindings and FLC frame/dimension metadata.
- Render-layer classification and default C3X/Civ III ownership.
- Unclassified files, unresolved atlas indices, missing animation references, and semantic/runtime gaps; the strict vanilla generation closes all four at zero.

The editable unit-art head start is `inventory/vanilla_conquests_to_civ6_units.json`, documented in `docs/civ3_to_civ6_unit_mapping.md`. It seeds all 93 standard Conquests `PRTO_*` records with a Civ VI ArtDef target, confidence, rationale, and vanilla fallback. This source-art proposal does not replace the required BIQ/runtime correlation and can be overridden per scenario later.

Use `--fail-on-unclassified` for taxonomy work. `--fail-on-unresolved` is the final M6.0 gate. On the installed vanilla tree it closes 112 effective files, 76 PCX atlases/contact sheets, 124 BIQ primary unit types, 144 selectable unit-art directories, 26 resources, and 14 terrain types with zero unknowns.

The tracked inputs are `inventory/vanilla_atlas_layouts.json`, `inventory/vanilla_conquests_biq_semantics.json`, and `inventory/runtime_selector_census.json`. `extract_biq_semantics.js` reproducibly regenerates the semantic snapshot from the installed BIQ and layered PediaIcons through the existing read-only editor parser. PNG contact sheets contain Firaxis pixels, so they are generated only under the ignored `inventory/generated/` tree and are never redistributed.

## Required Final Records

Each reachable visual selection must eventually record:

- Semantic category and stable BIQ identifier/name where one exists.
- Layered source path and effective search-path winner.
- Source sheet/FLC dimensions, cell geometry, authored cell count, selected index, and reachable runtime index set.
- Terrain adjacency/topology, culture, era, city size, resource class, unit action/direction, or other category-specific selectors.
- Companion shadow, mask, palette, sound, projectile, or effect relationships where applicable.
- Renderer ownership: `mapped`, `vanilla_fallback`, `not_map_rendered`, or `unreachable`.
- Evidence source and confidence: file format, BIQ record, decompiled selection rule, runtime draw census, or reviewed fixture.
- Day/night and season ownership, anchor/calibration data, and scenario override behavior.

Authored atlas capacity and runtime reachability are separate numbers. Unused cells remain documented as unreachable rather than disappearing from the report.

## M6.0 Gate (complete)

M6.0 is complete only when:

- Every vanilla Conquests BIQ terrain, resource, city style, unit type, and map-visible improvement/effect is represented.
- Every effective map-art atlas has verified cell geometry and an annotated indexed contact sheet generated locally.
- Unit INI/FLC actions and eight-direction behavior are correlated with BIQ unit types and runtime animation states.
- Representative runtime capture reaches every known selector family and produces no unknown file/index pairs.
- Every render responsibility, including fog, borders, labels, selection, cursor, minimap, and HUD, has a tested ownership/layering decision.
- Every record is classified as mapped, vanilla fallback, not map-rendered, or unreachable with evidence.
- Strict inventory generation is deterministic and passes with zero unclassified or unresolved records.

The gate is executable as `m6_0_inventory_contract` plus `m6_0_installed_vanilla_inventory`. Production M6 terrain coverage and all M7 category gates consume this ledger. They may add scenario records, but they may not redefine what “all vanilla art” means.
