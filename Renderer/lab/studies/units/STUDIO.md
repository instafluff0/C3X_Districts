# Independent unit studio

The user requested an unconstrained reference outside the existing renderer before
further attempts at Civ VI parity. `studio.py` is an independent NumPy/Pillow
triangle renderer. It does not load the C3X DLL, invoke the body API, inherit its
sprite fit, cache a game bitmap, or use its output tone compressor.

```sh
python3 -m Renderer.lab.studies.units.frame_probe
python3 -m Renderer.lab.studies.units.studio --size 1600 --yaw -45 --elevation 35
python3 -m unittest Renderer.lab.studies.units.test_studio
```

NumPy and Pillow are required. Inputs are the existing full-resolution material
DDS files, original first idle pose/palettes and the verified authored frames.
The five Warrior components retain 1,601 vertices and 2,096 triangles. One uniform
normalization sets anatomy height to one; the camera projects physical source
proportions without legacy canvas fitting or C3X's isometric projection.

Implemented: original base/tint, first LEAN normal map with authored tangents,
AO, the three-channel source-family cooked specular equation, a shared studio
key light and hemispherical fill, and a 1536px shadow map rasterized from actual
unit triangles. Texture level zero is bilinearly sampled for this deliberately
high-resolution reference. sRGB texture data is explicitly decoded to linear;
Pillow's equivalent UNORM BC-block entry point does not change the compressed
bits. Final color receives direct sRGB transfer, with no sharpening.

This applies the playbook's mountain lessons: preserve source form and material
information, avoid destructive fitting, and retain resolution until final
reconstruction. The source resolves cloth folds, leather seams, face detail and
weapon materials that are difficult to distinguish in the current small render.
That does not prove their eventual visibility at gameplay scale.

Local outputs under `lab/out/units/studio/` include the 1600px front reference,
`studio-front-preview.png`, `camera-study.png`, and `sampling-front-1to1.png`.
The last panel compares a direct 160px render against a 1600px render reconstructed
to 160px using a linear-light box and linear-light Lanczos filter. No rendered
pixels in this panel are enlarged. The high-resolution reconstruction has cleaner
edges and steadier texture detail; Lanczos is a comparison, not an approved game
filter or evidence of Civ VI's own reconstruction.

Lighting/camera are explicit studio choices. Exact source environment cube/SH,
second LEAN variance scale, extra-slot metalness intake, and source output
processing remain incomplete. This image is an independent diagnostic reference,
not a claim of Civ VI parity. The larger image also exposes source mesh overlap
and low-polygon faceting that should not be concealed with fabricated detail.

The retained city shader already carries the recovered dual-lobe specular
interpretation (`q8_city_direct_specular` in `native/city_fidelity/city.hlsl`).
The current unit shader still reduces that RGB texture to a scalar gloss and GGX
response. This is a concrete remaining material gap. Its perceptual contribution
must be isolated at matched size, camera, pose and lighting before promotion.

Installed `Base/ArtDefs/Camera.artdef` declares DEFAULT_CAMERA FOV 45 and a tilt
curve of 55 at time 0 and 45 at time 100, plus bloom/exposure settings. The local
`source-camera.json` records exact metadata and its hash. Engine interpretation
of tilt, active zoom and the screenshot camera are not established; these values
are not silently treated as a recovered camera matrix. The inspected DX11 shader
archive has no reflected resource name matching the tested tone/sharpen/FXAA/TAA/
SSAA/bloom/exposure terms. That negative inventory does not prove absence of
postprocessing in the engine.

Four analytic checks passed: sRGB roundtrip, texel centers/repeat/clamp sampling,
triangle barycentrics, and actual-triangle shadow direction. No production files,
DLLs, fixed references or injected code were changed by the studio experiment.

## Native Lab implementation

The independent studio findings now have an isolated native D3D11 Lab profile:

```sh
python3 Renderer/renderer.py lab units --case studio
python3 Renderer/renderer.py lab units --case studio-gameplay
python3 Renderer/renderer.py lab units --case studio-move
```

`lab_profile.py` builds a private, hashed source snapshot with guarded adapters
and a separate `UnitStudioLab` pack. `studio_response.hlsl` implements the studio
material equations. The pack retains source geometry, animations, UVs and
recovered tangent frames. The fixture uses anatomy fitting, the studio's physical
35-degree orthographic camera, positive-alpha tint, first-map normal detail,
full AO, the RGB dual-lobe specular response, the declared studio key/fill,
1536px triangle shadows, level-zero bilinear sampling, and direct sRGB output.
Four-times scratch dimensions plus the existing 4x MSAA resolve retain linear
color and coverage until final box reconstruction. No sharpening is applied.

Each capture contains all six subjects in three rows:

1. Original Lab fit, projection, lighting, shader, shadow resolution and sampling.
2. Old material under matched anatomy, studio camera/light, sampling and transfer.
3. Native studio material under those same matched conditions.

The second row is a material control, not a production screenshot: the old
material's brighter diffuse response can clip under studio light and direct
transfer. The first row supplies the actual original-method comparison.
Rearranged `before-after` and `matched-material` panels preserve captured pixels
at 1:1. Outputs and a production-isolation receipt live under
`lab/out/units/studio-native/`; the receipt records the candidate DLL hashes.

This is a port of the material and fidelity findings, not bitwise equivalence to
the independent rasterizer. Native triangle coverage/4x MSAA, pose-local height
shadow parameterization and bias differ from the CPU reference. Native ground
shadows use 0.16 display-space alpha as a neutral-backdrop approximation to the
CPU studio's 0.68 linear shaded-ground multiplier; terrain backgrounds therefore
are context checks, not recovered shared-environment lighting. Studio camera and
light settings remain explicit fixture choices. Missing second-map variance
constants, source environment illumination and metalness intake remain missing.

The implementation is Lab-only. It does not stage a DLL, modify production unit
bindings, replace references or alter game ownership/dirty bounds. The large
spear envelopes and production shared-lighting fit still require integration
work before any game promotion.


After game promotion, `studio` cases retain their pinned pre-promotion private
snapshot as the historical A/B fixture. Ordinary `detail` and `gameplay` Lab
cases render the current game implementation. The game adapts the recovered
material to its shared environment and authoritative map basis, so the fixed
studio fixture is not presented as a production-lighting baseline.

## Installed game close-up comparison

`python3 -m Renderer.lab.studies.units.game_closeup` renders the staged
`Renderer/bin/C3XRenderer.dll` directly with a private magnified Warrior pack.
It does not rebuild or replace the DLL or production assets. The private pack
uniformly scales vertex positions and palette translations by four, the authored
fit by 1.25, and uses the native 2x projection with a 512px base canvas. This
produces 10x geometric magnification, beyond the playable camera range, while
retaining texture bytes, normals, tangents, skinning, materials, 4x scratch
sampling, noon lighting and display transfer. Reversing the power-of-two
geometry scaling must reproduce the original payload bytes exactly.

Eight native headings pass the body-boundary guard. The comparison uses the
front view at heading 5 and the retained original studio image, reduced in
linear light to the same approximate colored-subject height. Game pixels are
neither enlarged nor sharpened. Their differing projection, viewpoint and
lighting remain visible; this is not a controlled material-only comparison or
a screenshot from a playable 10x zoom. Uniform world-size magnification also
retains the game's existing world-space shadow bias, rather than asserting
pixel equivalence to a hypothetical new camera implementation.

Outputs and hash receipts are under `lab/out/units/game-closeup/`.
The comparison resolves cloth, leather and weapon detail in the game path,
with visibly darker shading and less prominent highlights than the studio.
It supports investigating lighting/display response next; it does not prove
that any one shading parameter alone explains the difference. Production input
hashes are checked before and after rendering.

Lighting follow-up: `unit_body_renderer.h` takes the shared key direction but
normalizes color, intensity and ambient changes against noon, then applies
unit-side base coefficients. A global multiplicative sun adjustment can
therefore cancel in the unit's noon ratio. The native `LinearOutput` also
compresses linear RGB by `1 + max(R,G,B)` before display encoding; the retained
studio uses direct display encoding. The comparison does not isolate these
effects, but it establishes why simply increasing global ambient brightness
is not a reliable match. Calibrate shared key/fill/exposure and display response
with units, terrain and cities together while preserving the shared shadow
direction and day/night cycle.
