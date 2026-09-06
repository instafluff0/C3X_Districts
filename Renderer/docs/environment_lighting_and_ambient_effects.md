# Environment Lighting And Ambient Effects

## Purpose

All 3D-owned terrain, water, cities, wonders, districts, units, and effects consume one shared C3X environment state. Moonlight on water, illuminated windows, street lamps, furnaces, torches, flames, and smoke must not become unrelated category-specific day/night systems.

The environment is driven by C3X's authoritative hour and season. It changes presentation only and never advances time, weather, production, visibility, or gameplay.

## Milestone Ownership

- **M6.1 baseline (complete):** deterministic hour/season variation for production terrain and water.
- **M6.4 shared environment foundation:** sun/moon direction and color, sky/ambient contribution, exposure, shadows, water Fresnel/specular response, emissive material channels, light-activation curves, and generic ambient-effect attachment records.
- **M7.3 cities:** bind city windows, lamps, and other static emissive groups to city models. City-scale animated flames/smoke wait for or consume M7.5's effect implementation.
- **M7.5 effects:** render deterministic attached flames, smoke, sparks, steam, and similar looping effects using the existing absolute-time scheduler.
- **M9 natural wonders:** bind waterfalls, geysers, steam, volcanic glow/smoke, waves, mist, and other natural-wonder material/effect attachments.
- **M10 constructed wonders:** bind wonder-specific emissive groups and ambient-effect sockets, including construction/completion differences.
- **M11 districts:** bind district-base and building-attachment lights/effects. A newly represented building can add its own windows, lamps, chimneys, or flames without replacing the district base.

M6.4 proves the shared primitives with synthetic terrain/water and emissive fixtures. It does not take visual ownership of cities, wonders, or districts early. It follows M6.3 real-terrain expansion so environment and material extensions apply consistently across normalized packs.

## Frame Environment

Each immutable render frame derives these source-independent values from the captured C3X environment:

- Hour, optional transition fraction, season, and stable presentation timestamp.
- Sun direction, color, intensity, and shadow contribution.
- Moon direction, color, intensity, and shadow contribution when enabled.
- A shared stylized shadow projection: direction rotates with hour, but the
  projection slope remains fixed so caster height—not clock elevation—sets
  footprint length. Every raised renderer-owned class receives both
  normal-driven face shading and a neighboring cast shadow.
- Sky/ambient color, exposure, and tone-mapping parameters.
- Water reflection/specular scale, roughness response, Fresnel response, and optional normal/wave animation phase.
- Global emissive multiplier and a continuous night-activation value from `0.0` in daylight to `1.0` at full night.
- Optional wind/effect phase inputs only when a later visual effect actually needs them.

The moonlit-water proof does not require a literal reflected moon disk. Directional moonlight must produce a stable, camera-consistent specular response on water while land, coast geometry, retained Civ III overlays, and HUD remain correctly exposed.

## Emissive Materials

The generic pack material supports an optional emissive texture/mask, emissive color multiplier, intensity, and activation policy. Typical policies are `always`, `night`, `twilight-and-night`, and explicit C3X hour ranges. Activation fades use the environment's continuous night value where available; they do not require duplicate day and night models.

Static emissive changes do not require continuous redraw. They invalidate the map when C3X's hour/transition changes, then remain idle. Emissive surfaces participate in exposure and optional restrained bloom inside the 3D map render only; they must not color-grade or bloom Civ III's labels, fog, cursor, minimap, HUD, or other retained layers.

L13A proves only this shared activation contract with a non-visual diagnostic.
The first visible emissive/light presentation is L17, using actual city windows,
lamps, and analytic attachments. Earlier infrastructure such as L14 roads
inherits the environment without invented lights; later improvements may add
only their own source-backed attachments when their lab gates enter.

## Ambient Effect Attachments

A model or district-kit component may expose named source-agnostic attachment records:

```text
AmbientAttachment
  id
  effect_asset
  local_transform
  activation_policy
  intensity/scale
  stable_phase_seed
  visibility and optional state requirements
```

Examples include `city.blacksmith.chimney_smoke`, `district.industrial.furnace_flame`, and `wonder.lighthouse.beacon`. The source adapter may discover these from Civ VI VFX, Building, District, CityGenerator, or model metadata, but the compiled pack never stores Civ VI identifiers or formats as runtime behavior.

Animated flame, smoke, spark, steam, wave, or beacon effects use stable instance IDs and absolute presentation time. They request additional frames only while visible and active. Frame skipping advances directly to the correct pose; it does not queue catch-up frames or alter Civ III timing. Off-screen, hidden, daylight-disabled, paused, or fully static attachments request no continuous redraw.

## District And Building Composition

Emissive groups and ambient attachments belong to the district-kit component that visually creates them:

- The district base can own paths, common lamps, or a central brazier.
- A `by-building` Library attachment can add its own windows without changing Campus base lighting.
- A Factory or Power Plant attachment can add lit windows, smoke, sparks, or furnace glow when C3X reports that building present.
- A `by-count` preset contains the combined emissive/effect set for that count stage.

The complete visible kit is resolved before native suppression. If a required building attachment is missing and the district falls back atomically, its renderer-owned lights/effects are also suppressed so they cannot hover over the native 2D district.

## Civ VI Source Investigation

The Civ VI adapter should inspect `GameLighting.artdef`, `WaterMaterials.artdef`, `Wave.artdef`, `VFX.artdef`, `Buildings.artdef`, `Districts.artdef`, `CityGenerators.artdef`, model materials, and cooked package references. It should preserve available emissive maps, light groups, VFX identities, sockets, transforms, and activation metadata as evidence, then normalize useful results into generic C3X materials and attachments.

The installed-source evidence and repeatable metadata probe are documented in [civ6_lighting_findings.md](civ6_lighting_findings.md). Current evidence confirms a mixed asset/runtime system: declarative ArtDef curves select cooked light rigs; `Light.blp` and `VFX_FireFX.blp` expose named light/effect resources; `SHARED_DATA` supplies emissive/effect textures; and landmark packages appear to bind named instances. Typed package decoding is still required to recover exact parameters and transforms.

Missing Civ VI metadata is not a blocker for the runtime contract. A pack may author its own emissive mask or attachment, use a static approximation, or disable the effect. Firaxis payloads remain local prototype inputs and are not redistributed.

The offline adapter currently normalizes a deliberately small eight-texture
fire/glow/smoke/steam slice into `AmbientEffectsNormalized`. These are prepared
inputs, not runnable effects: the pack marks particle layout and Light/VFX/socket
binding unresolved, so no category may claim or display them until its normal lab
and integration ownership gate proves the missing behavior.

The companion analytic-light adapter decodes all six reflected parameters from
the Base `LightPackageEntry` records and normalizes 12 production-like resources
into `AnalyticLightsNormalized`; four test/negative entries are excluded by
policy. These records still have no model attachment binding and carry an
explicit unapproved-calibration status, so they likewise remain prepared inputs
rather than enabled lighting.

The later source-independent effect compiler supplies bounded authored small
fire, smoke, and steam graphs plus exact normal/reduced density policies. It
validates every referenced normalized texture, samples from absolute time and a
stable instance ID, and caps live particles. This closes the generic behavior
contract without claiming equivalence to the still-undecoded Civ VI resource
scripts. Category-owned socket binding and visual calibration remain at their
normal Lab/M7.5 gates.

## Verification

The shared environment gate includes deterministic noon, sunset, midnight, and sunrise renders at two viewport sizes and verifies:

- Sun/moon and exposure values are finite, continuous, and driven only by captured environment state.
- Water receives a visible but bounded directional moonlight response at night without becoming uniformly bright.
- Night emissive fixtures activate and daylight fixtures deactivate without swapping geometry.
- Static night scenes become idle; visible animated attachments request bounded frames through the normal Civ III redraw path.
- Stable effect IDs and phase seeds produce identical poses after skipped frames and replay.
- Missing emissive textures/effects degrade to non-emissive geometry or category fallback without suppressing unrelated content.
- Retained Civ III layers are neither relit nor bloomed.

Later city, effect, wonder, and district gates add representative attachment matrices and confirm that construction, completion, damage, abandonment, building additions, season changes, fallback, and config-off behavior activate or remove the correct lights/effects exactly once.

## M6.4 Implementation

M6.4 is complete. `native/environment_runtime.h` defines the source-independent frame environment, local transforms, bounds, activation policies, ambient attachments, and deterministic attachment results. `native/environment_runtime.cpp` evaluates continuous sun/moon, ambient, exposure, shadow, night/emissive activation, and bounded water Fresnel/specular values from captured C3X hour and season. The D3D terrain path consumes this shared state for all currently renderer-owned land and water without creating a clock or presenter.

The generic `c3x.material.v0` fixture in `samples/environment/m6_4_environment.fixture.json` demonstrates an optional emissive channel, analytic point light, and animated ambient attachment with an explicit local transform, state requirement, bounds, stable phase seed, and missing-resource policy. Its attachment phase is a pure function of absolute presentation ticks and seed. Static emissives contribute no animation count; invisible, inactive, missing, or native-fallback owners contribute no redraw request.

This milestone deliberately renders no replacement city, wonder, district, fog, label, selection, minimap, HUD, or UI. Those categories consume these primitives only at their own ownership gates.

## I13A Game Integration

I13A now uploads the same evaluated environment into the frozen approved
production terrain shader. Authoritative Civ III hour and season participate in
the complete-frame cache key; noon, sunset, midnight, and sunrise are distinct
and deterministic at both zooms. Raised terrain, volcanoes, forest/jungle,
shore bodies, and river clutter share one frame cast direction and normal-driven
self shading. Retained Civ III overlays are not relit, and the non-visual
emissive policy does not create object lights before L17/I17.
