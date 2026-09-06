# Natural Wonder Rendering Contract

## Purpose

Natural wonders are a standalone renderer category. They are not ordinary BIQ landmark terrain, generic mountains/volcanoes, constructed Great/Small Wonders, or ordinary C3X districts. Current C3X represents each placed natural wonder as a completed `NATURAL_WONDER_DISTRICT_ID` instance with an authoritative natural-wonder ID and tile coordinate. The renderer observes that state and replaces only its visual body.

Natural wonders are M9. Constructed wonders move to M10, and districts remain the final milestone as M11.

## Authoritative C3X Inventory

The inventory must parse the effective natural-wonder definition set using the same replacement precedence as C3X:

1. Scenario `scenario.districts_natural_wonders_config.txt`, when present.
2. User `user.districts_natural_wonders_config.txt`, when present and no scenario replacement applies.
3. `default.districts_natural_wonders_config.txt` otherwise.

The checked-in default currently supplies 18 `#Wonder` blocks, but that is only a mapping seed. C3X supports up to 64 definitions, dynamic replacement sets, and scenario-authored identities; tests must not hard-code 18 as complete runtime coverage.

Each effective definition and placed instance records:

- Stable natural-wonder ID/name and config provenance.
- Authoritative tile coordinate and pixel anchor from the C3X district instance.
- Required base terrain, optional adjacent terrain/river requirement, and adjacency direction.
- Native `img_path`, row, column, and any selected proxy/animation metadata needed for exact fallback and diagnostics.
- Up to eight configured animation alternatives, including hour mask, season mask, direction, frame time, and pixel offsets.
- Visibility/exploration/fog state and whether the retained native name label is eligible to draw.
- Visual orientation, bounds, depth envelope, environment state, and deterministic variant seed.

BIQ landmark terrain and volcano tiles remain M6 terrain selectors unless an actual C3X natural-wonder instance identifies the tile. A similar name or appearance never promotes a terrain tile into this category.

## Source Mapping

The Civ VI adapter inventories Base and DLC `Features.artdef`, the `NaturalWonders` collection in `TerrainStyle.artdef`, `Clutter.artdef`, `VFX.artdef`, `Water.artdef`, `WaterMaterials.artdef`, `Wave.artdef`, `Landmarks.artdef`, and referenced cooked packages. It maps C3X names to source candidates with `exact`, `approximate`, or `authored_required` status, separately reports unavailable installed targets, and retains evidence for model pieces, terrain materials, clutter, water integration, orientation, VFX, emissives, and lights.

The checked-in first pass is `Renderer/inventory/vanilla_c3x_to_civ6_natural_wonders.json`, documented in `Renderer/docs/civ3_to_civ6_natural_wonder_mapping.md` and enforced by `Renderer/inventory/natural_wonder_mapping_inventory.py`. It covers every default PCX cell and animation count, distinguishes exact matches from provisional stand-ins and authored requirements, and resolves preferred targets against installed `Features.artdef` definitions. It must remain editable conversion input rather than becoming a runtime identity table.

Civ VI multi-tile layouts are source-authoring information, not C3X gameplay state. A compiled natural-wonder kit may contain multiple visual pieces and may overhang neighboring tiles, but it is rooted at the single authoritative C3X instance anchor. It cannot create hidden gameplay tiles, suppress neighboring terrain/objects, change adjacency, or extend yields/passability. A future C3X multi-tile gameplay feature would require a new versioned scene contract.

The runtime pack contains only generic assets such as:

```text
natural_wonder/yosemite/base
natural_wonder/yosemite/terrain_blend
natural_wonder/yosemite/clutter
natural_wonder/angel_falls/waterfall
natural_wonder/yellowstone/steam_attachments
```

Runtime code never branches on Civ VI names, ArtDefs, BLPs, or source footprint conventions.

## Composition And Ownership

A natural-wonder kit can combine:

- Main mesh or terrain-integrated geometry.
- Base-terrain blend/decal and optional local material override.
- Directional or adjacency-specific pieces.
- Clutter/vegetation/scatter that belongs to the wonder.
- Water surfaces, waterfall sheets/particles, foam, spray, or wave response.
- Static emissive materials and analytic lights.
- Attached smoke, steam, fire, birds, mist, sparks, or other ambient effects.

M6 retains ownership of the underlying terrain and M7 retains its independently resolved infrastructure/resources unless the complete natural-wonder contract explicitly masks a visual subregion. Civ III retains fog/shroud, borders, grid, selection, labels, cursor, minimap, HUD, and UI. The existing C3X natural-wonder name label remains native and must stay correctly anchored, visible, and unlit above the replacement.

Replacement is atomic per instance. C3X suppresses the native natural-wonder sprite and its legacy custom animation only after the complete required kit is loaded and the renderer accepts the instance. Missing geometry, required directional pieces, device failure, or unsupported source state restores the complete native body exactly once; renderer-owned attachments disappear with it.

## Environment And Animation

Natural wonders consume M6.4's shared sun/moon, water, emissive, analytic-light, and ambient-attachment system. Waterfalls, geysers, steam, volcanic glow/smoke, waves, mist, and similar motion use M5.3 absolute presentation time and stable instance/effect IDs. Static scenes request no continuous redraw. Hidden, fogged, off-screen, paused, daylight-disabled, fallback, and completed one-shot effects contribute no active frame request.

C3X remains authoritative for hour and season. The renderer may vary snow, vegetation, water, emissives, and effect intensity without changing terrain identity, placement, visibility, yields, movement, or any gameplay rule.

## Verification Gates

### M9.1 Inventory And Mapping

- Parse default, user, and scenario replacement behavior with synthetic fixtures and report every effective definition and placed instance.
- Account for every field, native PCX cell, animation alternative, direction, terrain/adjacency condition, and retained label/fog responsibility.
- Generate a swappable C3X-to-source mapping ledger with confidence, Base/DLC provenance, source parts, footprint/orientation evidence, and unresolved dependencies.
- Prove that ordinary landmark terrain and volcanoes are not misclassified.

### M9.2 Rendering And Integration

- Render deterministic land, mountain, volcanic/geothermal, coastal, and water examples at two viewport sizes across noon, sunset, midnight, sunrise, and representative seasons.
- Cover static, animated, directional, adjacency-sensitive, source-multipart, fog-hidden, scrolling/wrapping, clipping/depth, missing-asset, reset, and config-off cases.
- Prove animated effects remain deterministic across skipped frames and static natural wonders become idle.
- Prove the native name label and all retained Civ III layers remain visible exactly once and are not relit or bloomed.
- Require every effective custom-on natural-wonder instance to resolve to a complete 3D kit or fail visibly without native body replay; partial replacement cannot pass.

Manual screenshots remain a single batched strategic checkpoint after automated scene, contact-sheet, image, ownership, and performance gates pass.

## Civ III Patch Dependencies

Start with existing C3X `natural_wonder_configs`, `district_tile_map`, `natural_wonder_info.natural_wonder_id`, `draw_district_for_tile`, the natural-wonder image/animation paths, and the existing retained-label draw hook. Extend the existing visible-scene capture and suppression logic if those fields are not yet exported.

No new Civ III function or `civ_prog_objects.csv` entry is currently known to be required. Audit a dedicated natural-wonder mutation/placement hook only if normal C3X map invalidation cannot expose create/load/replace transitions promptly; do not request an address speculatively.
