# Civ VI Lighting And Ambient-Effect Findings

## Result

Civ VI's night presentation is not solely programmatic. The installed asset tree contains declarative time-of-day curves, cooked global light rigs, named analytic lights, emissive and effect textures, scripted VFX resources, and apparent model-side light/effect attachments. The Civ VI engine still evaluates these records, interpolates time, executes VFX behavior, and renders the result, so C3X must convert useful source data into its own source-agnostic environment/material/attachment representation rather than trying to run Civ VI assets directly.

Run the repeatable metadata-only inventory from the C3X root:

```powershell
py Renderer\tools\asset_compiler\civ6_lighting_probe.py `
  --assets-root "Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets"
```

The command writes `civ6_lighting_probe.json`. Paths in that report are relative to the supplied Assets root; it reads ArtDef XML, filenames, file sizes, and printable package strings but extracts no cooked payload.

## Confirmed Source Locations

### Global Environment

`Base/ArtDefs/GameLighting.artdef` contains two profiles, `DEFAULT_LIGHTING` and `WONDER_TOD`. Both bind Sunrise, Noon, Sunset, and Night phases through `GameLighting` library entries in `default_lighting.xlp` and `lighting/default_lighting.blp`. Their `WeightCurve` records contain time/weight control points. The ArtDef also contains sun azimuth/tilt/zenith, sun color, exposure, light-map weight, fog-of-war tint, and cloud-shadow collections.

`Base/Platforms/Windows/BLPs/lighting/default_lighting.blp` is the referenced cooked package. Its printable metadata includes `Sunrise_LightRig`, `Noon_LightRig`, `Night_LightRig`, `m_vSunDirection`, and `m_vSunIntensity`. This is direct evidence that Civ VI's global lighting is data-backed; interpolation and shading remain engine behavior.

### Analytic Lights

`Base/Platforms/Windows/BLPs/Light.blp` contains named resources including `ApplyLightMapWeight`, `DL_OrangeGlow`, `DL_OrangeGranaryLight`, `DL_ArenaLight`, `DL_AQDBathGlow`, `DL_StepwellGlow`, `DL_VolcanoGlow`, `DL_YellowBoatLight`, and related variants. DLC trees contain additional `Light.blp` packages.

The Base package's reflected `LightPackageEntry` records are now decoded. Each
of its 16 entries contains exactly six typed fields: `Color`, `Radius`,
`Intensity`, `Attenuation`, `TimeOfDay`, and `ApplyLightMapWeight`. The offline
importer normalizes the 12 production-like lights into generic point-light
records and explicitly excludes four warning/test/negative fixtures. Color,
range, intensity, attenuation, activation, and light-map weighting are therefore
confirmed source parameters. Model attachment identity and C3X visual
calibration remain unresolved, so the pack is not runtime-enabled.

### Fire, Smoke, Glow, And Steam

`Base/Platforms/Windows/BLPs/VFX_FireFX.blp` contains concrete VFX identities and scripts such as `CommonFireScript`, `CommonSmokeScript`, `FireFXAnalyticLightSet`, `Block_Torch_Animated`, `Brazier_Fire_light`, `ChimneySmoke`, `BuildingSteam`, `FX_Light_Flicker`, lantern glow, stadium glow, solar-farm beams, power-plant steam, pillaged fire/smoke, sparks, and lit/unlit particle materials. Base and DLC `VFX_FireFX.blp` packages are therefore primary conversion inputs for animated ambient effects.

This package is not shaped like the compact Base Light library. Its CIVBLP
header exposes 64 big-data entries, and representative qualified script/resource
strings occur in that big-data region rather than as unique direct strings in
the reflected package-data stripe. Reusing the static landmark/light decoder
would therefore be false progress. VFX conversion needs a separate big-data
entry/chunk profile before script dependencies or sprite layout can be claimed.

Concrete texture payload candidates live under `Base/Platforms/Windows/BLPs/SHARED_DATA` and equivalent DLC paths. Examples include:

- `TEXTURE_FXt_Fire_Torch_ANI_01`
- `TEXTURE_firefx_fire_torch_animation`
- `TEXTURE_FX_4Flames`, `TEXTURE_FX_4Flames_dark`, and `TEXTURE_FX_4Flames_light`
- `TEXTURE_FX_Flame_Glow001` and `TEXTURE_FX_Flame_Glow002`
- `TEXTURE_FX_LightSource` and colored light textures
- `TEXTURE_FX_Lighthouse_Beam`
- `TEXTURE_WON_Lighthouse_E` and `TEXTURE_WON_Mont_St_Michel_Emissive_A`
- `TEXTURE_GDR_emissive`

The generated report inventories 450 filename candidates on the documented installation. The broad list is discovery evidence, not a claim that every filename is map-relevant or that its material role is proven.

The conservative offline intake now converts eight representative payloads into
the source-independent `AmbientEffectsNormalized` pack: a torch sheet, four-flame
sprite, flame glow, generic light-source glow, dust/smoke sheet, smoke cloud,
steam cloud, and sharp smoke mask. The importer preserves dimensions, mip counts,
DXGI format, color space, hashes, and generic usage hints. It deliberately marks
sprite layout, particle behavior, analytic-light parameters, and model attachment
bindings unresolved. The current normalized DDS set is 973,272 bytes and has no
runtime source-format dependency.

`Water.artdef`, `WaterMaterials.artdef`, and `Wave.artdef` are now parsed structurally for their `BLPEntryValue` bindings. The installed metadata confirms `Water/Coast`, `Water/Deep`, `Water/Lake`, `Water/River`, `Water/RiverSource`, and `WaveTest` entries in the `Water`/`Wave` classes. Exact cooked water shader parameters remain unresolved, so the M6.4 runtime uses its own bounded generic Fresnel/specular model rather than claiming parameter equivalence.

### Model-Side Attachments

Printable metadata in landmark packages repeats concrete light/effect instance names:

- `landmarks/city_buildings.blp`: torches, braziers, and many `FX_ChimneySmoke*` instances.
- `landmarks/tilebases.blp`: district light posts, braziers, building fires, and smoke/effect instances.
- `landmarks/hero_buildings.blp`: `DL_ArenaLight`, lighthouse/lantern names, theater fire/smoke, workshop fire, and related instances.

The strongest current interpretation is a layered graph:

```text
ArtDefs choose logical model/variant
  -> landmark/model BLP records attach named effects and lights
  -> Light.blp and VFX_FireFX.blp define reusable resources
  -> SHARED_DATA supplies texture payloads
  -> Civ VI engine evaluates curves, scripts, transforms, and rendering
```

The city component decoder now resolves `AttachmentPointList` to typed
`AttachmentPointCookData` arrays. Each 32-byte city record names exactly one
skeleton bone, whose normalized rest transform provides the exact local socket.
The current 44-component proof subset contains 35 such sockets: 28 smoke, five
flame, one night-light, and one unresolved semantic name. Resource identity,
typed light/VFX parameters, and full state behavior remain unresolved.

The same decoder confirms the 64-byte material record's `Generic_Emissive` slot
at offset `0x3c`. The proof subset converts 96 emissive material bindings, and
the base city package exposes 24 distinct emissive textures. Night activation
is a C3X runtime policy applied to that confirmed channel, not a claim about the
source engine's activation algorithm.

The mine/farm proof extends that material decoder to a recursively composed
improvement subset. Its 93 normalized components contain 108 confirmed static
emissive material bindings. The same attachment table also carries nested
component records, which are distinct from true effect sockets; 35 effect
resource identities remain unresolved and are intentionally not fabricated in
the improvement preview. These findings support night-weighted static
emissives, but not inferred particles, analytic lights, or source-engine state
behavior.

The resource static importer now uses the same confirmed material offset. Five
uranium rock variants resolve `TEXTURE_RES_Uranium_E` from the
`Generic_Emissive` slot and normalize it as a generic night-activated emissive
mask. This is a proven static resource-lighting input; it is not evidence for an
analytic light, particle effect, or source-engine bloom value.

## C3X Conversion Target

The importer should preserve evidence in provenance, then emit only generic pack data:

- Environment profiles and continuous hour curves.
- Material emissive mask/color/intensity and activation policy.
- Analytic point/spot/directional light records with local transforms and bounds.
- Ambient effect assets plus model attachment socket/transform, visibility/state conditions, and stable phase seed.
- Water material inputs needed for bounded sun/moon specular and Fresnel response.

Static emissives should redraw only when C3X's hour/environment changes. Flames, smoke, steam, animated water, and rotating/flickering beacons use M5.3's absolute-time dirty scheduler only while visible. Missing effect metadata can fall back to non-emissive geometry, an authored C3X approximation, or the complete native category draw.

## Supported M6.4 Vertical Slice

The checked-in probe now emits a machine-readable `supported_vertical_slice` with evidence strength attached to each relationship. Structured GameLighting and water ArtDef bindings plus named Light/VFX resources and shared effect-texture filenames are confirmed source metadata. City material emissive slots and city attachment socket-to-bone local transforms are also confirmed. Exact typed resource identity, color, radius, falloff, particle behavior, and state binding remain unresolved and are not inferred.

Accordingly, the supported slice converts the confirmed resource classes into the generic C3X contract while using explicit authored fixture values and transforms. It proves the import boundary and runtime behavior without pretending to reproduce Civ VI's proprietary shader or attachment graph and without redistributing cooked payloads.

Run the conservative texture conversion locally with:

```sh
python3 Renderer/tools/asset_compiler/ambient_effect_texture_importer.py
python3 Renderer/tools/asset_compiler/analytic_light_importer.py
```

Its tracked mapping contains source-specific names only on the offline side. The
generated manifest exposes generic IDs such as `effect/fire/torch_sheet` and
`effect/smoke/cloud`; later M7.5 work may bind those only after particle layout,
blend mode, timing, and socket ownership are proven or explicitly authored.

## Remaining Extraction Work

1. Decode `VFX_FireFX.blp` resource/script dependencies into a supported generic subset; eight representative textures are normalized, but sprite layout and unsupported Civ VI shader behavior still need proven metadata or an authored approximation.
2. Resolve city and landmark attachment sockets to exact Light/VFX resource identity and state/visibility conditions; city socket transforms and the Base light parameters are already decoded.
3. Extend the confirmed city/resource/improvement `Generic_Emissive` material slot to remaining package families and distinguish embedded payloads from `SHARED_DATA` references.
4. Investigate `WaterMaterials.artdef`, `Wave.artdef`, and their cooked packages for moonlit-water inputs.
5. Calibrate imported analytic-light values for C3X only in the owning Lab gates; typed source parameters do not by themselves establish visual equivalence.

No new Civ III function or `civ_prog_objects.csv` entry is required for this source-discovery work.
