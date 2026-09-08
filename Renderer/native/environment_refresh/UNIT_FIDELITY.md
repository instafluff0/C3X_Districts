# Generic unit source fidelity in Game Integration

The prior runtime retained older normalized normals and a simple diffuse shader.
The Lab source changes alone did not update the already compiled roster payloads.
This refresh replays the source evidence for every imported runtime component and
uses the selected BeautyStudies unit material response in the native sprite path.

## Selected paths and their production equivalents

- `terrain_lab/v2/systems/objects/beauty_objects.cpp`: selected material frame,
  full-resolution DDS sampling and material metadata. Production uses native
  scene time, sprite anchors and existing pose-local shadow visibility.
- `terrain_lab/v2/shaders/objects/beauty_objects.hlsl`: exact GGX, sky-weighted
  ambient, AO, gloss, emissive and rim equations. `prepare_unit_shader.py`
  extracts these equations into `unit_shader.h`; `unit-shader-provenance.json`
  pins both source and generated output. Multiplicative owner-color modulation
  retains texture detail, with native civilization color and existing mask policy.
- `tools/asset_compiler/compound_landmark_importer.py` and
  `unit_model_extractor.py`: replay the proven octahedral normal decoder, exact
  UV0, source primitive addressing and mesh-local skin palette remap. These Lab
  implementations were already present; `refresh_unit_normals.py` applies them
  to 375 active imported mesh primitives from five normalized source packs.
- `terrain_lab/v2/tools/build_beauty_objects.py`: its per-material repeat/clamp
  contract is carried into each native animation part. Sharing a texture does
  not share its sampler state. Both address axes are represented independently.
- The associated importer and BeautyStudies tests, `docs/unit_asset_conversion.md`
  and `audits/objects/WARRIOR_SOURCE_PASS.md` remain the detailed source authority.
  No Warrior-specific shader, unit-name test or role-based sampler rule is added.

The offline normal refresh produces a separate `UnitNormalFidelity` catalog.
`prepare_units.py` publishes `UnitAnimationFidelity`, retaining all 78 normalized
unit entries and 94 standard Conquests native aliases. Only the normal fields in
animation vertices change; all other vertex bytes, indices and animation palettes
remain exact. Unchanged textures are hardlinked, not duplicated or recompressed.
The runtime's default catalog is `UnitAnimationFidelity`; `INSTALL.bat` keeps its
existing staging behavior and needs no new invocation or configuration.

## Rendering and cache contracts

- Use source vertex normals, transformed by the existing inverse-transpose
  skinner and renormalized. Preserve original procedural assets' own normals.
- Preserve positions, UV0, primitive order, source attachments, native uniform
  fit scale and local palette remaps. Do not rescale individual axes.
- Load full DDS dimensions and mip chains. Use 16x anisotropy, zero mip bias,
  native sprite resolution and existing 4x MSAA, as in the isolated unit path.
  The natural-scene 2x/-1 settings do not silently replace the unit witness.
- Bind base color, AO and gloss from material metadata; optional emissive is
  supported by the same generic contract. Current roster has no emissive maps.
  Preserve paired LEAN files offline but do not load or guess their decode.
- Pin every referenced material channel with its action inside the existing
  96 MiB payload budget. Preserve the 8 MiB sprite cache and its identity,
  dirty bounds, action cursor, native timing, fallback and retained terrain.
- Log dimensions, mip count, first mip, address mode and channel on texture
  loads through the normal debugger-output path. No new game file logging.

## Verification

See `verification/environment_refresh/unit-checkpoint.json` for release hashes
and actual results. The complete-roster gate exercises both game zoom scales,
all available actions, four facing directions and three action cursor positions,
plus the nine-family direction sheets, interruption/config-off/clip/cache checks.
The byte gate covers 2,459 payload pairs and 474 unmodified DDS chains; all 375
imported components resolve to fingerprinted source normals. The independent
NumPy versus production C++ normal test covers 393 component poses in humanoid,
civilian, mounted, mechanical, aircraft and naval families, with maximum error
about 2.12e-7. The pre-existing all-action evaluator also passes 4,622 sampled
poses and 1,611 attachment checks with the refreshed normal pack.

## Explicit limits and remaining work

The isolated Warrior r4 image is a close-up with a separate Civ V noon LUT.
Native sprites retain the existing linear-to-display transfer, game exposure,
native projection and authoritative sun/moon phase. This is a material/data
port, not a claim of pixel-identical lighting or of equal visible detail at a
roughly 56-pixel native unit height. The selected source LEAN evaluation remains
unresolved. No sharpening filter or invented texture detail is applied.

The known intermittent frozen-profile boundary-cache comparison and the frozen
L19A fixture-hash failure remain visible in broader workflow evidence. The staged
old DLL reproduces the boundary failure. No frozen witness or tolerance changes.

The natural source-fidelity-r13 profile remains the default. The separately
prepared static-water/reflection candidate is still opt-in; the new city layout,
actual palace geometry, facade light buffer and city/palace night-lighting port
are not included in this unit release. Their selected Lab validators pass, but
production implementation, matched controls and performance gates remain work.
No new patch rows, injected source changes, INSTALL execution or game launch.
