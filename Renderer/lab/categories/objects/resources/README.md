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
texture slots. Production keeps `ResourceNormalized` until a batch is approved.

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

Category commands prepare the static bundle and animation pack when their source
inputs or compilers change. Current inputs are `ResourceNormalized` and
`ResourceAnimatedLab`, with clip units in `Renderer/lab/shared/resources/clip_units.json`.
Builders use disposable output and preserve source bytes, animated root deltas,
marine school facing and the accepted fish surface offset. Historical runtime
enablement/checkpoint flags and release numbers are not workflow state.
