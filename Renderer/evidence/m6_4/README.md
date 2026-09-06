# M6.4 Shared Environment Evidence

M6.4 adds a source-independent shared environment and ambient-attachment foundation without changing Civ III injection code or claiming ownership of later map-object categories.

## Runtime evidence

- `Renderer/native/environment_runtime.h` defines continuous frame-environment values, generic local transforms and bounds, activation policies, and ambient attachment input/output records.
- `Renderer/native/environment_runtime.cpp` derives sun, moon, ambient, exposure, shadow, night/emissive activation, and bounded water Fresnel/specular values solely from captured C3X hour and season. Attachment phase is derived from absolute presentation ticks plus a stable phase seed.
- `Renderer/native/c3x_renderer.cpp` applies the shared environment to currently renderer-owned terrain and coast/sea materials. It creates no window, presenter, clock, or game loop.
- `Renderer/native/native_smoke.cpp` renders noon, sunset, midnight, and sunrise at 320x200 and 640x480, asserts a spatially directional but bounded moon response on water, proves a static night emissive remains idle, and proves animated attachment determinism and missing/hidden fallback.

## Generic pack evidence

`Renderer/samples/environment/m6_4_environment.fixture.json` is a source-agnostic `c3x.material.v0` fixture with an emissive channel, analytic point light, and flame attachment. The attachment has an explicit local transform, bounds, activation policy, state requirement, stable phase seed, period, and missing-resource policy. It also records that fog, labels, selection, minimap, HUD, and UI remain Civ III-owned.

`Renderer/environment/contract.py` validates that generic fixture and evaluates activation, absolute phase, visibility/state requirements, static idle, and explicit degradation. No source-specific file format or installed path appears in the runtime fixture.

## Local source evidence

`Renderer/tools/asset_compiler/civ6_lighting_probe.py` structurally parses installed GameLighting, Water, WaterMaterials, and Wave ArtDef bindings and inventories named Light/VFX/SHARED_DATA/landmark metadata without extracting cooked payloads. `Renderer/docs/civ6_lighting_probe.json` records a conservative vertical slice:

- ArtDef bindings and printable package resource names are `confirmed`.
- Model-resource attachment relationships inferred from repeated names are `inferred`.
- Exact cooked colors, ranges, falloff, sockets, and transforms remain `unresolved`.
- The generic fixture therefore uses explicit authored values rather than inventing source parameters.

## Gates and ownership

`m6_4_environment_runtime` is the portable executable gate; `civ6_lighting_metadata_local` is the installed metadata-only gate. M6.4 required no new Civ III function, address, injected-state field, or `civ_prog_objects.csv` entry. All M7 objects, effects, cities, units, retained overlays, fog, labels, minimap, HUD, and UI remain native until their own milestones.
