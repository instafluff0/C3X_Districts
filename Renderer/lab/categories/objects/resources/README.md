# Resources

Current normalized resource bodies and southeast-facing animation; retain source family limitations.

Land studies show Iron, Cattle, Horses, Wheat, Gold and Dyes. Water studies show
Fish and Whales. Both have detail and surrounding-terrain cases. This is a small
go-to sample, not a claim that every source resource is replaced in C3X.

`roster` places all 26 vanilla Conquests resources, by their BIQ display names,
on grassland with Fish/Whales offshore; `relief` puts every land resource on a
hill. Their native log lists `ROSTER ... replaced=0|1` as a census, not a gate:
unreplaced resources keep Civ III's sprite in game but are blank in Lab.
`native-minerals` places each mineral on every terrain Conquests allows it
(`Renderer/lab/studies/resources/native_terrains.json`), including mountains;
`native-minerals-alternates` shows Lab-only `Name~label` alternates beside them.
`python3 Renderer/lab/studies/resources/audit.py --label NAME` renders every case at
128 and 256 and writes per-resource crops beside the original Civ III sprite to
`Renderer/lab/out/resources/audit/NAME/index.html`.

Reworked resources are previewed from a candidate pack before promotion:
`--resource-pack ResourceCompositionLab` rebuilds it with
`tools/asset_compiler/build_resource_compositions.py` and renders through the
same production renderer (`C3X_RENDERER_RESOURCE_PACK`). Its sources come from
`tools/asset_compiler/resource_composition_sources.py`, which imports the Civ VI
ArtDef entries the profiles name (Base and DLC), keeps each rock's authored
burial and records terrain/feature variants. Every size, count, extra sink,
decal choice and terrain layout comes from
`Renderer/inventory/resource_composition_profiles.json`; the runtime only places
baked instances, matches resource names by data aliases and picks a variant for
the tile's terrain. Ground decals are flat `decal/` meshes with soft alpha;
model and decal textures are block-copied into atlases for the eight resource
texture slots.

Production resources are these compositions: the `resources` preparation job bakes
`Renderer/packs/ResourceCompositions` without Lab alternates, and the runtime
loads it by default (`--resource-pack` still selects a candidate for review).
`ResourceNormalized` remains a source pack for the animation runtime. Animated
subjects in a composition (`animated/<binding>`, including derived schools such
as `fish.few` from `school_orientation.thin_school`) come from
`ResourceAnimationRuntime`. The game compiles its shaders from the pinned
`Renderer64ResidentRuntime` pack; `Renderer/tools/overlay_resource_shading.py`
carries the current resource shading into it (review with `--dry-run`; `--match TEXT`
keeps other features' unaccepted shader edits that mention resources out of the game).

A tile's resource name selects a body only through pack aliases, matched
exactly and case-insensitively. A composition alias wins. Otherwise the name
must be listed under `"aliases"` in `ResourceAnimationRuntime/bindings.json`,
which `build_resource_animation_runtime.py` writes from the profiles' Civ III
names. A name with neither has no body; no legacy static groups remain. The
runtime loads only the animation bindings those aliases or the compositions
name. `native/test_runtime_resource_names.py` fails on any resource or unit
name literal in the Renderer64 DLL sources.

The fixed references preserve the earlier black-backdrop defect around animated
resources. Current code fixes guarded scene-linear backdrop accumulation.
Comparison will show that intended difference; it does not block Integration.
Replace the fixed reference only if the user wants the corrected appearance to
become the new visual comparison point.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

Category commands prepare the composition pack and animation pack when their
inputs or compilers change. Composition inputs are the profiles and
`ResourceCompositionSources`; animation inputs are `ResourceNormalized` and
`ResourceAnimatedLab`, with clip units in `Renderer/lab/shared/resources/clip_units.json`.
Builders use disposable output and preserve source bytes, animated root deltas,
marine school facing and the accepted fish surface offset. Historical runtime
enablement/checkpoint flags and release numbers are not workflow state.

The runtime never keys behaviour on resource names, because scenarios bring
their own resources. An animation binding with `"medium": "water"` swims its
`submersion` (tiles) below the surface of a coast, sea or ocean tile. Renderer64
draws it in the underwater layer that `sandbox/water_surface.hlsl` composites,
without a ground shadow. On a marsh the same binding stays at the surface.
Animated bodies stand upright on their anchor's ground like units. Their depth
uses the highest ground under their footprint (`footprint_ground`), so rising
ground behind an animal never hides a leg.

Resource bodies take the profiles' `look` (gain, saturation; the pack's
`look.json`), as units take theirs. The user chose gain 1.0 with no extra
saturation on 2026-10-21. In game, look values must be judged through
Renderer64's filmic display, which the Lab tone map does not reproduce. Use
`scripted_game_test.ps1 -Scenario resources` with
`-RendererOptions C3X_RENDERER_RESOURCE_LOOK=gain,saturation` to compare.
`overlay_resource_shading.py --match game_resource_look` carries the look into
the pinned pack. The marine blend in `water_surface.hlsl` has no "resource" text,
so copy that file by hand.
