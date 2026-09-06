# C3X Custom Rendering Configuration Contract

## Status

This is the implemented v0 contract. `Renderer/definitions/definition_parser.py` parses it into a deterministic intermediate catalog; table-driven tests cover the syntax, layer merge, diagnostics, references, and path safety before injected code relies on it.

The complete starter fixture is `Renderer/samples/config/default.custom_rendering.txt`.

## Design Principles

- Human-editable text consistent with C3X's existing `key = value` and `#Section` style.
- One format can describe every map-rendered category without one enormous C structure.
- Names are preferred for authoring; numeric IDs and sprite metadata remain available for exact matching.
- Scenario definitions and art resolve through Civ III's normal scenario search path.
- Missing or invalid 3D definitions fall back to Civ III by default.
- Source formats never appear in runtime rules. A rule references a normalized C3X pack asset ID.
- Definitions are deterministic. There is no filesystem enumeration order or random selection without an explicit seed rule.

## Files And Precedence

Renderer definitions are loaded in this order:

1. `default.custom_rendering.txt` from the C3X folder.
2. `scenario.custom_rendering.txt` through `BIC_get_asset_path` for the active scenario.
3. `custom.custom_rendering.txt` from the C3X folder.

This mirrors C3X's main `default.c3x_config.ini -> scenario.c3x_config.ini -> custom.c3x_config.ini` chain. A custom file has final preference authority; a scenario can supply a complete art set without modifying C3X.

Each section has a required stable `id`. If a later layer defines the same section type and ID, it replaces the complete earlier section. Full-section replacement is preferred over field-by-field merging because it is easier to inspect and test. A later layer may use `disabled = true` to remove an inherited definition.

The parser records source file, layer, and line for every active definition.

## Path Resolution

Asset packs are declared explicitly:

```text
#Pack
id   = base_world
path = mod:Renderer\packs\BaseWorld

#Pack
id   = scenario_world
path = scenario:Art\CustomRendering\MyScenario
```

Prefixes:

- `mod:` resolves relative to the C3X folder.
- `scenario:` resolves using the active Civ III scenario search path.
- `file:` is an explicit local development path and is disabled in distributable mode by default.

Pack paths and any paths inside manifests must be normalized and prevented from escaping their pack root. Renderer rules reference asset IDs, not arbitrary model paths.

## Section Types

The v0 grammar uses the same style as `default.tile_animations.txt`: a directive starts a section and following `key = value` lines populate it.

- `#Profile`: global renderer policy and category ownership.
- `#Pack`: named pack root.
- `#Asset`: optional alias or calibrated presentation metadata for a pack asset.
- `#Rule`: Civ III metadata selector mapped to an asset.
- `#Environment`: day/night and seasonal lighting/material policy.

Unknown keys are errors with file and line. Unknown section types are errors. Comments and blank lines are ignored.

## Implemented Parser Details

- Section and key names are case-sensitive. Enumerated values such as `replace`, `terrain`, seasons, and booleans are accepted case-insensitively and normalized to lowercase.
- A full line enclosed in `[` and `]` is a comment, matching the starter fixture. Blank lines are ignored. Inline comments are not part of v0.
- Stable IDs may contain letters, digits, `.`, `_`, and `-`; IDs are case-sensitive.
- Duplicate keys and duplicate section-type/ID pairs within one layer are errors. A definition in a later layer may reuse the pair and completely replaces the earlier section.
- A disabled definition requires only `id` plus `disabled = true`; it removes the matching inherited section and remains in the catalog's disabled-definition audit list.
- Integers, finite floating-point values, booleans, RGB triples, seasons, and hour/range lists are converted to typed catalog values. Invalid or unknown values are never preserved as untyped strings.
- `#Asset asset` is a logical asset ID within its named pack, not a filesystem path. After merging, asset-to-pack, rule-to-asset, and profile-to-environment references must all resolve.
- `mod:` and `scenario:` pack paths must be relative and remain under their configured roots after normalization. `file:` is available only when explicit local-development mode is enabled; a relative `file:` path resolves from the declaring definition file.
- Paths inside pack manifests must be relative to and remain inside the pack root. The parser exposes the same root-escape validator for manifest loaders.

Every active catalog definition records the declaring file, section line, layer name/index, and declaration index. Parse and merge failures carry structured diagnostics with file, line, section type, section ID when known, key, message, and expected form.

## Profile

```text
#Profile
id                 = default
terrain            = replace
features           = civ3
roads               = civ3
rivers              = civ3
improvements        = civ3
resources           = civ3
cities              = civ3
units               = civ3
effects             = civ3
missing_asset       = fallback
world_seed          = map
environment         = earthlike
```

Category values:

- `civ3`: Civ III owns the category.
- `replace`: matching 3D content suppresses the corresponding Civ III base draw.
- `augment`: 3D content is added but Civ III's base draw remains.
- `capture-only`: collect and log scene data without changing visuals.

`missing_asset = fallback` is required as the v0 default. Development profiles may select `warn` or `error` without changing release behavior.

## Assets And Calibration

Pack manifests own raw models/materials. `#Asset` adds Civ III presentation metadata without modifying the source pack:

```text
#Asset
id             = grassland.base
pack           = scenario_world
asset          = terrain/grassland/base
anchor_x       = 0.50
anchor_y       = 0.50
scale          = 1.00
offset_x_px    = 0
offset_y_px    = 0
fit_width_px   = 128
fit_height_px  = 64
casts_shadow   = true
receives_shadow = true
```

Object assets such as units and cities normally anchor at projected bottom-center. Terrain assets normally anchor at tile center. Fit dimensions are calibration guides, not clipping rectangles.

## Rules

A rule identifies a map-rendered category, zero or more selectors, and one result:

```text
#Rule
id                = terrain.grassland.default
category          = terrain
priority          = 100
terrain_type      = grassland
asset             = grassland.base
variant_selection = coordinate-hash
replacement       = replace
```

More specific Civ III art matching is allowed:

```text
#Rule
id           = terrain.grassland.sheet2.sprite0
category     = terrain
priority     = 200
terrain_type = grassland
sheet_index  = 2
sprite_index = 0
asset        = grassland.sheet2.sprite0
```

Representative object rules:

```text
#Rule
id            = resource.horses
category      = resource
resource_name = Horses
asset         = resource.horses
show_in_seasons = spring, summer, fall, winter

#Rule
id            = city.european.ancient.town
category      = city
culture_group = European
era           = Ancient Times
city_size     = town
asset         = city.european.ancient.town

#Rule
id            = unit.rome.legionary.run
category      = unit
civilization  = Romans
unit_type     = Legionary
action        = run
asset         = unit.rome.legionary
animation     = run
```

## Selector Vocabulary

All selectors are optional except `category`. M2 may implement them incrementally, but unknown selectors must fail loudly rather than being ignored.

Shared selectors:

- `map_x`, `map_y`, `landmark`, `owner`, `civilization`, `era`.
- `show_in_day_night_hours`, using 0..23 values and inclusive ranges such as `18-5`.
- `show_in_seasons`, using summer, fall/autumn, winter, spring.
- `adjacent_to`, using the existing tile-animation direction vocabulary.

Terrain and feature selectors:

- `terrain_type`, `real_terrain_type`, `sheet_index`, `sprite_index`.
- `pcx_file`, `pcx_index`, `river_mask`, `road_mask`, `railroad_mask`.
- `has_forest`, `has_jungle`, `has_marsh`, `has_pollution`, `has_crater`.
- `improvement`, `terrain_building`, `coast_shape`, `neighbor_mask`.

Resource selectors:

- `resource_name`, `resource_id`, `resource_class`, `pcx_index`.

Resource replacement ownership is map-only. A resource `replace` rule may suppress the matching native map body/shadow after its complete replacement is ready, but it never replaces `resources.pcx` globally or changes Civilopedia, city-screen, trade-network, advisor, diplomacy, notification, or other non-map icons. The seed mapping and curatable exceptions are in `civ3_to_civ6_resource_mapping.md`.

City selectors:

- `culture_group`, `era`, `city_size`, `has_walls`, `is_capital`, `city_style_index`.

Unit selectors:

- `unit_type`, `unit_id`, `unit_class`, `direction`, `action`, `fortified`, `hit_point_band`.

M7.4 will version this vocabulary rather than silently loosening the v0 parser. Its planned unit selectors include `current`, `active`, `selected`, `moving`, `stack_display_index`, `event_kind`, and scenario-defined action names. Timing fields such as event ID, start time, duration, and normalized progress are captured runtime inputs used to evaluate an animation; they are not generally selectors and must not create one rule match per frame. See `runtime_animation_and_frame_pacing.md`.

For the initial M7.4 contract, unit category ownership applies to the animated body only. A unit `replace` rule must not suppress or replace Civ III's selection cursor/ring, health bar, left-side activity/status marks, stack indicators, or related unit HUD. Those overlays remain Civ III-owned until a later, explicitly approved and versioned ownership extension.

M9, M10, and M11 will version the vocabulary with separate `natural_wonder`, `wonder`, and `district` categories. Planned natural-wonder selectors include stable C3X natural-wonder ID/name, required terrain, adjacent terrain/river and direction, native image row/column, configured animation identity/direction, hour/season, and visibility state. Their rule result references one logical natural-wonder kit rooted at the authoritative C3X anchor; source multipart/footprint data remains pack metadata rather than gameplay state. Planned constructed-wonder selectors include BIQ improvement ID/name, Great/Small class, construction/completion state, owner, map placement, and alternate orientation. Planned district selectors include stable district ID/name/type, culture, era, `render_strategy`, effective building count, dependent building ID/name, construction/damage/abandonment state, coastline orientation, and connection/topology masks. District rule results reference a logical kit, base, attachment, count-stage preset, or topology piece; they never expose Civ VI file formats to runtime code. See `natural_wonder_rendering.md` and `wonder_and_district_rendering.md`.

The contract deliberately keeps both semantic selectors and actual selected-art metadata. Semantic matching is friendly to scenarios; sheet/sprite matching lets C3X preserve Civ III's topology and variant decisions precisely.

## Composite Tile Resolution

A tile is not mapped to one giant replacement asset. The scene builder emits independent terrain, feature, infrastructure, resource, city, unit, and effect instances rooted at authoritative Civ III anchors. Rules resolve each instance and may inspect neighboring tile metadata when needed.

For example, "grassland sprite 5 immediately east of mountain sprite 2, containing a culture-index-4 medium city" can activate separate rules for:

- The grassland base material/mesh variant.
- A western mountain-edge transition or blend treatment.
- The mountain's own geometry variant on the neighboring tile.
- The city's culture, era, and size model.
- Seasonal asset/material variants.
- Lighting and emissive behavior for the current C3X hour.

This is how the system can approximate a 1:1 visual match while still benefiting from continuous terrain and a depth-buffered 3D scene. Exact Civ III indices remain selectors, but a rule may intentionally translate them into procedural topology instead of a literal one-file replacement.

## Rule Selection

For each captured item:

1. Reject rules whose category or selectors do not match.
2. Choose the highest explicit `priority`.
3. If tied, choose the rule with the most matched selectors.
4. If tied, choose the rule from the higher-precedence config layer.
5. If tied, choose the rule declared later in that layer.

This ordering is deterministic and must be covered by tests. A rule may define `variant_selection = coordinate-hash`, which hashes map coordinate, world seed, and rule ID. It must not use frame time or enumeration order.

### Implemented Resolver Details

`definitions/rule_resolver.py` consumes only the typed, merged M2.1 catalog and one captured metadata record. It never opens a pack, model, or texture. An optional set of available asset IDs lets callers report missing compiled assets without turning rule matching into asset loading.

- Config-off, `civ3`, and `capture-only` ownership return before candidate matching or asset availability checks.
- Selector names and values retain their typed M2.1 representation. String selector comparisons are case-insensitive, including Windows PCX names; stable definition and asset IDs remain case-sensitive.
- `adjacent_to` accepts either one captured adjacency value or a captured collection. Wrapped hour ranges include both endpoints, so `18-5` matches 18 through 23 and 0 through 5. `autumn` and `fall` are the same normalized season.
- Specificity is the count of matched selector fields. It excludes `category`, `priority`, `asset`, `animation`, `replacement`, `variant_selection`, and `disabled` control/result fields.
- Ranking uses exactly `(priority, specificity, layer index, declaration index)`. Each matched loser reports the first losing stage; rejected candidates report every mismatched selector with expected and actual values.
- Coordinate variants use the first 64 bits of SHA-256 over rule ID, map X, map Y, and world seed in a fixed encoding. The output is a stable seed for a later pack-variant modulo operation, independent of frame time and enumeration order.
- Missing winners and unavailable winning assets return an explicit Civ III fallback. The diagnostic always reports zero asset-payload loads and distinguishes availability checks from payload access.

For standalone inspection, serialize the M2.1 merged catalog and captured item metadata as JSON, then run:

```powershell
py -m Renderer.definitions.rule_resolver catalog.json item.json --world-seed 42
```

The emitted `c3x.renderer_rule_resolution.v0` record contains the winner, effective replacement mode, rank components, variant seed, every candidate explanation, and fallback reason when applicable.

## Day/Night And Seasonal Environment

```text
#Environment
id                    = earthlike
day_night_source      = c3x
season_source         = c3x
sunrise_hour          = 6
sunset_hour           = 18
sun_azimuth_degrees   = 135
noon_sun_color        = 255, 244, 220
midnight_ambient_color = 22, 30, 52
night_exposure        = 0.35
shadow_quality        = medium
seasonal_materials    = true
moonlight_enabled     = true
water_moon_specular   = 0.45
emissive_night_scale  = 1.0
bloom_strength        = 0.08
```

The existing C3X cycle modes determine the hour and season. The environment maps those values to continuous lighting. Rule filters and pack material variants handle discrete art changes.

The renderer must support mixed ownership. For example, 3D terrain receives native night lighting while 2D resources continue using C3X's current `Art/DayNight/...` images. The bridge must suppress old proxy replacement only for a category fully owned by 3D, preventing double application. Environment vocabulary is versioned by the M6.4 contract; the fields above are planned author-facing names, not permission for an older parser to silently accept them.

## Pack Contract

The current `c3x.asset_pack.v0` prototype will evolve into a validated, source-agnostic manifest. It should contain:

- Stable asset IDs and type (`terrain`, `model`, `material`, `animation`, etc.).
- Normalized coordinate system, scale, bounds, anchor hints, and units.
- Model and texture paths relative to pack root.
- Material channels and color-space metadata.
- Optional emissive mask/color/intensity and `always`, `night`, `twilight-and-night`, or explicit-hour activation policy.
- Source-agnostic analytic lights and ambient-effect attachments with local transforms, state/visibility requirements, bounds, and stable phase seeds.
- Named animations and deterministic variants, with clip duration, playback rate, loop/completion policy, transition/blend metadata, and named fallback chains where applicable.
- Optional season variants and renderer capability requirements.
- Provenance and redistribution policy excluded from runtime branching but retained for tooling.

The runtime must never need ArtDefs, BLP, FGX, CivNexus6, Blender, or import provenance to draw a pack.

## Error And Fallback Contract

- Parse errors report file, line, section ID, key, and expected value.
- Missing pack or asset references report the winning rule and attempted resolved path.
- Release behavior falls back to Civ III for the affected item/category.
- One invalid rule must not disable unrelated categories.
- Config-off mode performs no renderer asset loading.
- A debug dump can explain which rule won, which selectors matched, and why candidates lost.

## Required M2 Tests

- Default, scenario, and user precedence.
- Complete-section replacement and `disabled = true`.
- Invalid section/key/value diagnostics with line numbers.
- Path prefix resolution and pack-root escape rejection.
- Rule priority, specificity, layer, and declaration-order tie breaks.
- Terrain type plus exact sheet/sprite matching.
- Named resource, city, and unit matching.
- Wrapped day/night ranges and season filters.
- Deterministic coordinate variants.
- Missing asset fallback and config-off no-load behavior.
