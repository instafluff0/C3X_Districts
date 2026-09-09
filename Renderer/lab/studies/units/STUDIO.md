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
