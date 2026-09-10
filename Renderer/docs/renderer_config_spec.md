# C3X Custom Rendering Configuration Contract

## C3X map effects and cache switches

The C3X configuration loader accepts these booleans in `custom.c3x_config.ini`
(and the normal scenario configuration layers):

```ini
enable_custom_rendering_reflections = false
enable_custom_rendering_waves = false
enable_custom_rendering_cache = true
```

Reflections and waves default to `true`; setting either to `false` intentionally
omits that effect. The cache defaults to `false` in distributed defaults and is
enabled in this checkout's local custom configuration. Effects are independent
of cache enablement. They do not affect config-off native rendering or units.
The cache switch selects the existing bounded ring-four world-region method;
it does not implement instant cold-map jumps or asynchronous native presentation.

The September 9 navigation production candidate also uses this existing switch
for the verified retained-wave and animation-backdrop paths, exact backdrop
dependencies, indexed animation receivers and larger bounded cache tiers.
When `C3X_RENDERER_WORLD_REGIONS=1`, absent newer environment options inherit:
`THREE_ZOOM_MEMORY=1`, `WORLD_BACKDROPS=1`, `WORLD_WAVES=1`,
`BACKDROP_DEPENDENCIES=1`, `COMPOSITION_RECEIVER_INDEX=1`, and
`UNIT_POSE_MEMORY=512` (each with the `C3X_RENDERER_` prefix).
Explicit standalone overrides still win, including zero. Without the cache
switch these options retain their smaller/off defaults. The injected bridge's
existing waves and reflection switches remain authoritative. This DLL policy
does not require another injected patch or installation when the current cache
switch is already installed. Staging and verification status are recorded in
[the continuation](navigation_continuation.md).

The selected ceilings are 64 MiB viewport images, 832 MiB linear/depth animation
backdrops and 512 MiB unit-pose pixels / 4,096 entries. They are conditional
retention limits, not upfront reservations or a total process-memory allowance.
Other owners and the remaining live-game memory verification are described below.

The injected bridge applies these settings through the existing renderer
environment controls before DLL initialization. They take precedence over the
evaluation launcher and inherited effect/cache settings, and require a game
restart after editing. Install the updated C3X injected source to recognize the
new flags; the already staged renderer DLL supports them without rebuilding.
The two effect flags alone do not enable custom rendering.

The user-requested navigation evaluation profile is
`Renderer/run_navigation_evaluation.cmd`. It scopes experimental environment
settings to one launched game process; ordinary launches use the installed
cache/effect switches and the DLL defaults above. It enables world-region reuse, receiver-scoped shadow dependencies,
tighter natural bounds and current-capture ring four, while preserving waves
and reflections. It does not install C3X or change the configuration format.
See [the handoff](navigation_handoff.md) for measured coverage and limitations.

## Status

This is the implemented v0 contract. `Renderer/definitions/definition_parser.py` parses it into a deterministic intermediate catalog; table-driven tests cover the syntax, layer merge, diagnostics, references, and path safety before injected code relies on it.

The complete starter fixture is `Renderer/samples/config/default.custom_rendering.txt`.

The separate C3X boolean `enable_custom_rendering_zoom` enables stepped main-map
camera levels on repeated `Z` presses when `enable_custom_rendering` is also on.
It is an integration setting rather than a pack-definition key; see
[the custom zoom contract](custom_rendering_zoom.md).

The base modern-machine tier budgets GPU tile geometry at 768 MiB,
CPU natural/ground retention at 96 MiB, viewport bitmaps at 32 MiB and static
resource/wave backdrops at 128 MiB. These are separate ceilings, not a total
process-memory allowance. Shadow atlases, textures, scratch targets, unit data,
snapshots, publications and driver allocations require additional space. Civ III
remains a 32-bit process; large physical RAM does not remove its address-space
limit. These changes were authorized for modern computers and remain subject
to widest-view and live-game memory verification.

`C3X_RENDERER_GPU_GEOMETRY_MIB` is a compile-time pressure-test override
(192–1024 MiB), not a definition-file setting. The older large-cache benchmark
macro also increases CPU/bitmap/backdrop tiers; identify it separately from the
production tier. Opt-in standalone diagnostics include `C3X_RENDERER_PROFILE=1`,
`C3X_RENDERER_REGION_SIZE=256`, `512`, or `2240`, and `C3X_RENDERER_BLOCK_CLIP=1`.
The `2240` experiment uses 2240-by-256 strips, with four-pixel main-pass
guards and another four pixels for reflections. The linear targets retain
2x resolution and MSAA4; their width and height are bounded independently.
`C3X_RENDERER_BOUNDED_POST=1` additionally limits that strip path's glow
dispatch and final color conversion to the guarded rectangle copied out.
Dispatch origins align to eight pixels to retain the complete-target
workgroup sampling. This switch is off by default; accumulated passes,
512-pixel siblings and the dynamic 128-pixel backdrop path retain complete
reconstruction. It does not reduce scratch allocation or MSAA resolve size.
`C3X_RENDERER_WORLD_RASTER_GRID=1` keeps regions anchored to captured world
coordinates for the scrolling parity experiment; it is off by default.
Region size defaults to 128; larger regions remain experimental. They retain
the same MSAA, reconstruction and filter quality, and use separate scratch
from the 128-pixel dynamic backdrop path. No configuration here enables native
asynchronous presentation.

`C3X_RENDERER_WORLD_REGIONS=1`, together with the world raster grid and
128-pixel regions, enables the experimental completed-region cache. It retains
BGRA output in worker-owned GPU textures, with 256 MiB for images, 96 MiB for
key/node metadata and at most 4,096 entries. `C3X_RENDERER_REGION_METADATA_MIB=32`
selects the lower pressure-test limit. These are additional conditional cache
ceilings, not total process or deferred-driver memory measurements. Admission
failure skips caching while preserving ordinary full-quality rendering.
`C3X_RENDERER_WORLD_REGIONS_CONTROL=1` renders the same full guarded regions
independently for comparison. Both paths preserve current publication ownership;
native surfaces are never retained by this cache. The feature defaults off.

`C3X_RENDERER_REGION_RECEIVER_SHADOWS=1` experimentally narrows completed-region
shadow dependencies from all casters in a required atlas page to the subset whose
projected bounds reach any current region receiver bounds. A four-shadow-texel
margin preserves normal offset, floor/tap footprint and raster tolerance. It does
not alter atlas rendering or the shared shadow shader. Required page coordinates,
selected caster versions/material bindings/offsets, reflection receivers and other
region dependencies remain represented. The evidence runner exposes
`--region-receiver-shadows`; the switch defaults off pending matched parity and
broader edit/visibility checks.

`C3X_RENDERER_TIGHT_NATURAL_BOUNDS=1` uses extrema of actual natural-mesh vertices
in the native affine projection, instead of projecting the corners of the mesh's
3D bounding box. The fixed-size retained metadata has no vertex copies or resource
ownership. Projection preserves the existing two-pixel culling margin; unsupported
tile aspect ratios use the previous bounds. The option affects culling and region
dependencies, not vertices, shading or shadow atlas contents, and defaults off.
The evidence runner exposes `--tight-natural-bounds`.

`C3X_RENDERER_REGION_INPUT_RING=4` extends foreground support from the default
two-tile ring to four tiles, using only full-appearance PREFETCH records in the
current capture. Topology-only records and older captures remain ineligible.
This stabilizes region contributor sets at the cost of preparing and retaining
more detailed geometry. The runner exposes `--region-input-ring {2,4}`. It adds
no capture permissions and does not synthesize off-screen appearance.

`C3X_RENDERER_COMPOSITION_RECEIVER_INDEX=1` reuses that current static assembly's
conservative spatial index for animation shadow-receiver selection. Exact
intersections and original ordering still determine receivers; foreign/posed
buffers and unsupported queries use the complete scan. It adds no retained
owner. The evidence runner exposes `--composition-receiver-index`; it is off
outside the production cache profile.

For the production cache profile, the default described above is enabled.
`C3X_RENDERER_BACKDROP_DEPENDENCIES=1` validates retained linear color/depth
against the exact current static-region dependencies. A bitmap-only region hit
cannot satisfy a backdrop request. Ordered contributors, lights, reflection,
content and geometry identities still decide reuse; missing proof redraws the
original backdrop. It follows the same production cache-profile default.

`C3X_RENDERER_UNIT_POSE_MEMORY=1` selects 256 MiB and at most 4,096 entries in
the existing exact unit-pixel cache; value `512` selects 512 MiB with the same
entry cap. Explicit other values keep 8 MiB / 128 entries; an absent option
follows the production policy above. The runner exposes
`--unit-pose-memory` and optional `--unit-pose-memory-mib 512`. Budget reductions
evict least-recently-used owners before lookup. These are cached pixel-capacity
limits; metadata, current output, temporary admission copies, loaded assets and
GPU resources are additional owners. Allocation/admission failure retains the
current completed body. Native directed action cursors, individual ambient
phases, quality, dimensions and cache keys do not depend on the memory tier.

`C3X_RENDERER_REGION_DIAGNOSTICS=1` emits per-region dependency component
fingerprints and world/screen rectangles for offline miss analysis. The evidence
runner exposes `--region-diagnostics`; `analyze_region_dependencies.py` classifies
misses and optionally compares earlier region pixels with `--images`. Diagnostic
hashes never authorize reuse or replace the full value key. Trace-heavy runs are
not performance evidence. This explicit mode raises the bounded trace-file limit
from 8 MiB to 32 MiB. Fingerprint or screenshot equality in one fixed-clock
sweep does not establish a safe general invalidation rule.

Region validity includes contributing draw order and transforms, material/content
and environment state, required shadow-page identities, reflected contributors
and nearby city lights. The current implementation derives those keys from
prepared geometry; it does not yet bypass geometry preparation after mesh eviction
or provide whole-map/disk-backed preparation. Newly exposed or differently captured
regions can still miss. Exact new-camera parity does not establish instant cold
jumps, animated performance, native presentation or the complete memory envelope.

Static shadow caster preparation is shared for the lifetime of one animation
composition by default. `C3X_RENDERER_COMPOSITION_CASTERS_CONTROL=1` restores
independent preparation for every region for matched timing/pixel comparisons.
The existing bounds/selection budgets and geometry ownership are unchanged.

`C3X_RENDERER_WATER_COVERAGE=1` is an opt-in standalone experiment. It omits
bed/water layers only when every uploaded flat-grid hydrology distance is
safely positive, and skips reflection work in regions with no remaining water
coverage. Shoreline, negative, uncertain and nonfinite samples retain the
existing passes. This switch is off by default pending measured pixel parity.

`C3X_RENDERER_WORLD_BACKDROPS=1` enables a separate animation-backdrop
experiment when the world raster grid is also enabled. Regions retain their
world-relative placement across pans; static scene, ownership, lighting,
topology revision, zoom and device identity still invalidate cached linear
color/depth. The 128 MiB backdrop cap is unchanged. Set
`C3X_RENDERER_BACKDROP_REUSE_CONTROL=1` to redraw each region independently
for pixel comparisons while preserving the same region placement. Both
controls are off outside the production cache profile; native presentation is unaffected.

`C3X_RENDERER_WORLD_WAVES=1` enables a bounded pool of immutable coast-cell
occurrences, including cached empty cells. Its GPU buffers have a 32 MiB cap;
metadata has a separate 16,384-entry cap. The complete requested cell set is
pinned before eviction. Raw wrapped coordinates retain their existing shadow
and world placement. Topology revision, map/wrap dimensions, zoom/target,
content and device changes invalidate the pool; light/time updates use the
unchanged dynamic shader inputs. `C3X_RENDERER_WAVE_REUSE_CONTROL=1` rebuilds
cells for independent comparisons. The new cell-local projection remains
experimental. The legacy path retains its 16 MiB active-buffer cap, but now
reports failure on overflow instead of silently dropping remaining ribbons.

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
